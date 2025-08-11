
import os
import urllib.parse
import redis
from redis.commands.search.query import Query
from azure.identity import UsernamePasswordCredential
from typing import List

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.redis import Redis
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains.reorder import LongContextReorder
from langchain.docstore.document import Document

# --- Project-specific Imports ---
from models.conversation import CCChatRequest, CCConfig
from models.models import ChatLinkBase
from common.logs_util import LogsUtil
from memorydb.auth import ElastiCacheIAMProvider, MemoryDB as RawMemoryDBClient
# Import the refactored data management functions
from . import utils

_logger = LogsUtil("FID.AI fastapi -> PersistentRAGAgent").get_logger()

class PersistentRAGAgent:
    """
    Manages a stateful, session-based RAG lifecycle by coordinating with utility
    functions that handle the direct data operations.
    """
    def __init__(self, chat_request: CCChatRequest, chat_link_data: ChatLinkBase, ccconfig: CCConfig):
        self.chat_request = chat_request
        self.chat_link_data = chat_link_data
        self.model_provider = chat_request.model_provider
        
        # --- Configuration from ccconfig ---
        self.index_name = ccconfig.index_name
        self.cc_memdb_user = ccconfig.cc_memdb_user
        self.cluster_name = ccconfig.cluster_name
        self.cluster_endpoint = ccconfig.cluster_endpoint
        self.region = "us-east-1"

        # --- Initialize Embeddings ---
        self.embeddings = self._init_embeddings()

        # --- Setup MemoryDB Connection ---
        self.redis_url, self.raw_client = self._get_memorydb_connection()
        
        # --- Initialize Vector Store ---
        self.vectorstore = self._get_or_create_vectorstore()

    def _init_embeddings(self):
        """Initializes the correct embedding model based on the model provider."""
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ.get("AZURE_OPENAI_API_BASE")
        azure_cred = UsernamePasswordCredential(
            client_id=os.environ.get('AZURE_OPENAI_CLIENT_ID'),
            username=os.environ.get('AZURE_OPENAI_SERVICE_ACCOUNT'),
            password=os.environ.get('AZURE_OPENAI_CLIENT_PASSKEY')
        )
        openai_token = azure_cred.get_token(os.environ.get('AZURE_OPENAI_COGNITIVE_API'))
        os.environ['AZURE_OPENAI_API_KEY'] = openai_token.token
        
        if self.model_provider == "open-ai":
            _logger.info("Initializing AzureOpenAIEmbeddings")
            return AzureOpenAIEmbeddings(
                azure_deployment=os.environ.get("AZURE_OPENAI_EMB_MODEL"),
                api_version=os.environ.get("AZURE_OPENAIEMB_API_VERSION")
            )
        elif self.model_provider == "bedrock":
            _logger.info("Initializing BedrockEmbeddings")
            return BedrockEmbeddings(region_name=self.region, model_id='amazon.titan-embed-text-v2:0')
        else:
            raise ValueError("Invalid model provider specified.")

    def _get_memorydb_connection(self):
        """Establishes connection to MemoryDB using IAM auth."""
        _logger.info(f"Connecting to MemoryDB cluster '{self.cluster_name}' with user '{self.cc_memdb_user}'")
        try:
            creds_provider = ElastiCacheIAMProvider(user=self.cc_memdb_user, cluster_name=self.cluster_name, region=self.region)
            auth_credentials = creds_provider.get_credentials()
            redis_url = f"rediss://{urllib.parse.quote(auth_credentials[0], safe='')}:{urllib.parse.quote(auth_credentials[1], safe='')}@{self.cluster_endpoint}:6379"
            raw_client = RawMemoryDBClient.from_url(redis_url, ssl=True, skip_full_coverage_check=True, decode_responses=True)
            if raw_client.ping():
                _logger.info("✅ Raw MemoryDB Client Connection Successful!")
            else:
                raise ConnectionError("Raw MemoryDB Client Ping Failed.")
            return redis_url, raw_client
        except Exception as e:
            _logger.error(f"❌ Error establishing MemoryDB connection: {e}")
            raise

    def _get_or_create_vectorstore(self) -> Redis:
        """Initializes the Redis vector store, creating the index with a session-based schema if it doesn't exist."""
        try:
            self.raw_client.ft(self.index_name).info()
            _logger.info(f"Index '{self.index_name}' already exists. Connecting.")
            return Redis.from_existing_index(
                embedding=self.embeddings,
                index_name=self.index_name,
                redis_url=self.redis_url,
            )
        except redis.exceptions.ResponseError:
            _logger.info(f"Index '{self.index_name}' not found. Creating new index with session and de-duplication support.")
            # Schema is now defined in the utility but the agent ensures it's created.
            schema = {
                "tag": [{"name": "user_id"}, {"name": "session_id"}, {"name": "doc_hash"}],
                "text": [{"name": "source"}],
            }
            return Redis.from_documents(
                documents=[],
                embedding=self.embeddings,
                index_name=self.index_name,
                redis_url=self.redis_url,
                index_schema=schema,
            )

    def add_documents_for_session(self, user_id: str, session_id: str, documents: List[Document], ttl_seconds: int = 14400):
        """Delegates adding documents to the utility function."""
        return utils.add_documents_for_session(
            raw_client=self.raw_client,
            vectorstore=self.vectorstore,
            index_name=self.index_name,
            user_id=user_id,
            session_id=session_id,
            documents=documents,
            ttl_seconds=ttl_seconds
        )

    def get_retriever_for_session(self, session_id: str, user_docs: List[Document], k: int = 5):
        """Creates a hybrid retriever for a specific session, filtering by session_id."""
        _logger.info(f"Creating SOTA retriever for session: '{session_id}'")

        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': k, 'filter': Redis.tag("session_id") == session_id}
        )
        _logger.info("   - Dense (vector) retriever configured with session filter.")

        bm25_retriever = BM25Retriever.from_documents(user_docs)
        bm25_retriever.k = k
        _logger.info("   - Sparse (BM25) retriever configured.")
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        _logger.info("   - Ensemble (hybrid) retriever configured.")

        reordering_compressor = LongContextReorder()
        final_retriever = ensemble_retriever | reordering_compressor
        _logger.info("✅ Final retriever created with re-ranking.")
        
        return final_retriever

    def session_has_documents(self, session_id: str) -> bool:
        """Delegates checking for documents to the utility function."""
        return utils.session_has_documents(self.raw_client, self.index_name, session_id)

    def delete_documents_for_session(self, session_id: str):
        """Delegates deleting documents to the utility function."""
        utils.delete_documents_for_session(self.raw_client, self.index_name, session_id)
