
import os
import urllib.parse
import redis
from redis.commands.search.query import Query
from azure.identity import UsernamePasswordCredential

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.redis import Redis
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains.reorder import LongContextReorder
from langchain.docstore.document import Document

from models.conversation import CCChatRequest, CCChatResponse, GraphState, CCConfig
from models.models import ChatLinkBase
from common.logs_util import LogsUtil
from memorydb.auth import ElastiCacheIAMProvider, MemoryDB as RawMemoryDBClient

_logger = LogsUtil("FID.AI fastapi -> RAGAgent").get_logger()

class RAGAgent:
    """
    Manages the entire RAG lifecycle for a multi-tenant application
    using a single Redis index, aligned with the project's structure.
    """
    def __init__(self, chat_request: CCChatRequest, chat_link_data: ChatLinkBase, ccconfig: CCConfig):
        self.chat_request = chat_request
        self.chat_link_data = chat_link_data
        self.model_provider = chat_request.model_provider
        
        # Configuration from ccconfig 
        self.index_name = ccconfig.index_name
        self.cc_memdb_user = ccconfig.cc_memdb_user
        self.cluster_name = ccconfig.cluster_name
        self.cluster_endpoint = ccconfig.cluster_endpoint
        self.region = "us-east-1" 

        # Initialize Embeddings 
        self.embeddings = self._init_embeddings()

        # Setup MemoryDB Connection
        self.redis_url, self.raw_client = self._get_memorydb_connection()
        
        # Initialize Vector Store
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
        """Establishes connection to MemoryDB using IAM auth, returning both URL and raw client."""
        _logger.info(f"Connecting to MemoryDB cluster '{self.cluster_name}' with user '{self.cc_memdb_user}'")
        try:
            creds_provider = ElastiCacheIAMProvider(user=self.cc_memdb_user, cluster_name=self.cluster_name)
            auth = creds_provider.get_credentials()
            
            redis_connection_string = f"rediss://{urllib.parse.quote(auth[0], safe='')}:{urllib.parse.quote(auth[1], safe='')}@{self.cluster_endpoint}:6379"
            
            raw_client = redis.from_url(redis_connection_string, decode_responses=True)

            if raw_client.ping():
                _logger.info("✅ Raw MemoryDB Client Connection Successful!")
            else:
                raise ConnectionError("Raw MemoryDB Client Ping Failed.")
            
            return redis_connection_string, raw_client
        except Exception as e:
            _logger.error(f"❌ Error establishing MemoryDB connection: {e}")
            raise

    def _get_or_create_vectorstore(self) -> Redis:
        """Initializes the Redis vector store, creating the index with a specific schema if it doesn't exist."""
        try:
            self.raw_client.ft(self.index_name).info()
            _logger.info(f"Index '{self.index_name}' already exists. Connecting.")
            return Redis.from_existing_index(
                embedding=self.embeddings,
                index_name=self.index_name,
                redis_url=self.redis_url,
            )
        except redis.exceptions.ResponseError:
            _logger.info(f"Index '{self.index_name}' not found. Creating new index.")
            # Multi-tenant schema 
            schema = {
                "tag": [{"name": "user_id"}],
                "text": [{"name": "source"}],
                "numeric": [{"name": "page"}],
            }
            return Redis.from_documents(
                documents=[], # Start with an empty index
                embedding=self.embeddings,
                index_name=self.index_name,
                redis_url=self.redis_url,
                index_schema=schema,
            )

    def add_documents_for_user(self, user_id: str, documents: list[Document]):
        """Adds documents to the index, tagging each chunk with the user_id."""
        _logger.info(f"Adding {len(documents)} document chunks for user: '{user_id}'")
        
        for doc in documents:
            doc.metadata["user_id"] = user_id
        
        keys = [f"{user_id}:{doc.metadata.get('source', 'unknown')}:{doc.metadata.get('page', 0)}:{i}" for i, doc in enumerate(documents)]
        keys_added = self.vectorstore.add_documents(documents, keys=keys)
        _logger.info(f"✅ Successfully added {len(keys_added)} chunks to index '{self.index_name}' for user '{user_id}'.")
        return keys_added

    def get_retriever_for_user(self, user_id: str, user_docs: list[Document], k: int = 5):
        """
        Creates a state-of-the-art hybrid retriever for a specific user.
        This retriever will only see data belonging to that user.
        """
        _logger.info(f"Creating SOTA retriever for user: '{user_id}'")

        # 1. Dense Retriever (Vectors) with user_id filtering
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': k, 'filter': Redis.tag("user_id") == user_id}
        )
        _logger.info("   - Dense (vector) retriever configured with user filter.")

        # 2. Sparse Retriever (Keywords)
        bm25_retriever = BM25Retriever.from_documents(user_docs)
        bm25_retriever.k = k
        _logger.info("   - Sparse (BM25) retriever configured.")
        
        # 3. Ensemble Retriever (Hybrid Search)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        _logger.info("   - Ensemble (hybrid) retriever configured.")

        # 4. Re-ranker
        reordering_compressor = LongContextReorder()
        
        final_retriever = ensemble_retriever | reordering_compressor
        _logger.info("Final retriever created with re-ranking.")
        
        return final_retriever
