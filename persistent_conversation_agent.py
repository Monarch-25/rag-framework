
import os
import boto3
from botocore.config import Config
from azure.identity import UsernamePasswordCredential
from langgraph.graph import MessagesState
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain.docstore.document import Document

# --- Project-specific Imports ---
from models.conversation import ChatRequest, ChatResponse, GraphState, CCConfig
from models.models import ChatLinkBase
from common.logs_util import LogsUtil
# Import the new stateful RAG agent
from doc_persistent_rag.persistent_rag_agent import PersistentRAGAgent
from chunking_strategies import get_chunker

_logger = LogsUtil("FID.AI fastapi -> PersistentConversationAgent").get_logger()

class PersistentConversationAgent:
    """
    Orchestrates a stateful, multi-turn conversation where document context
    persists throughout a user's session.
    """
    def __init__(self, chat_request: ChatRequest, chat_link_data: ChatLinkBase, chat_response: ChatResponse, graph_state: GraphState, user_files_content: str, is_file_data: bool, is_talk_to_data: bool, is_talk_to_doc: bool, ccconfig: CCConfig, chunking_strategy: str = "kamradt_modified"):
        # --- Standard Model and Client Initialization ---
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ.get("AZURE_OPENAI_API_BASE")
        self.azure_cred = UsernamePasswordCredential(
            client_id=os.environ.get('AZURE_OPENAI_CLIENT_ID'),
            username=os.environ.get('AZURE_OPENAI_SERVICE_ACCOUNT'),
            password=os.environ.get('AZURE_OPENAI_CLIENT_PASSKEY')
        )
        self.openai_token = self.azure_cred.get_token(os.environ.get('AZURE_OPENAI_COGNITIVE_API'))
        os.environ['AZURE_OPENAI_API_KEY'] = self.openai_token.token
        
        self.model_provider = chat_request.model_provider
        self.chat_request = chat_request
        self.openai_chat = None
        self.bedrock_model = None
        self.invalid_model_provide_msg = "Invalid model provider"

        if self.model_provider == "open-ai":
            self.openai_chat = AzureChatOpenAI(azure_deployment=self.chat_request.endPoint, api_version=os.environ.get('AZURE_OPENAI_API_VERSION'), max_tokens=4096)
        elif self.model_provider == "bedrock":
            boto3_config = Config(connect_timeout=5, read_timeout=600, retries={'max_attempts': 3})
            bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1", config=boto3_config)
            self.bedrock_model = ChatBedrock(model_id=self.chat_request.endPoint, client=bedrock_runtime, model_kwargs={"temperature": self.chat_request.temperature, "top_p": self.chat_request.top_p, "max_tokens": 4000}, verbose=False)
        else:
            raise Exception(self.invalid_model_provide_msg)

        # --- State and Configuration ---
        self.chat_response = chat_response
        self.graph_state = graph_state
        self.user_files_content = user_files_content
        self.is_file_data = is_file_data
        self.chat_link_data = chat_link_data
        self.is_talk_to_data = is_talk_to_data
        self.is_talk_to_doc = is_talk_to_doc
        self.ccconfig = ccconfig
        self.chunking_strategy = chunking_strategy

    def talk_to_doc(self, state: MessagesState):
        """
        Handles the 'talk to document' flow using a persistent RAG pipeline.
        Documents are indexed only once per session.
        """
        state["current_node"] = "talk_to_doc"
        self.chat_response.persona = "talk_to_doc"
        
        session_id = self.chat_link_data.session_id
        user_id = self.chat_link_data.user_id # Assuming user_id is available
        query = state["messages"][-1].content

        # Instantiate our powerful, stateful RAGAgent
        rag_agent = PersistentRAGAgent(chat_request=self.chat_request, chat_link_data=self.chat_link_data, ccconfig=self.ccconfig)
        
        try:
            # 1. Check if documents are already indexed for this session
            if not rag_agent.session_has_documents(session_id):
                _logger.info(f"No documents found for session '{session_id}'. Indexing now.")
                # 1a. Prepare and chunk documents
                docs = [Document(page_content=self.user_files_content, metadata={"source": "uploaded_document"})]
                text_splitter = get_chunker(self.chunking_strategy, rag_agent)
                chunks = text_splitter.split_documents(docs)
                _logger.info(f"Document split into {len(chunks)} chunks for session '{session_id}'.")

                # 1b. Add chunks to vector store with session tracking and TTL
                rag_agent.add_documents_for_session(user_id=user_id, session_id=session_id, documents=chunks)
            else:
                _logger.info(f"Documents for session '{session_id}' are already indexed. Skipping indexing.")

            # 2. Get the specialized, secure retriever for the session
            # Note: We still need the document content for the BM25 retriever.
            # In a real scenario, this might be cached or handled differently.
            temp_docs_for_bm25 = [Document(page_content=self.user_files_content)]
            retriever = rag_agent.get_retriever_for_session(session_id=session_id, user_docs=temp_docs_for_bm25)
            
            # 3. Invoke the retriever to get relevant context
            retrieved_docs = retriever.invoke(query)
            retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            _logger.info(f"Retrieved {len(retrieved_docs)} chunks for query in session '{session_id}'.")

            # 4. Generate a response using the retrieved context
            system_prompt = f"""You are an AI assistant. Use the following retrieved context to answer the user's question. The context is from a document the user uploaded earlier in this session. If the answer is not in the context, say so clearly.
            
            Context:
            {retrieved_context}
            """
            system_message = SystemMessage(content=system_prompt)
            
            # 5. Call the LLM with the refined prompt
            if self.model_provider == "open-ai":
                response = self.openai_chat.invoke([system_message] + state["messages"])
            elif self.model_provider == "bedrock":
                response = self.bedrock_model.invoke([system_message] + state["messages"])
            else:
                raise Exception(self.invalid_model_provide_msg)

            state["messages"].append(response)
            self.chat_response.text_content = response.content

        except Exception as ex:
            self.graph_state.failed_nodes.append("talk_to_doc")
            _logger.error(f"Exception in persistent talk_to_doc: {ex}")
            raise
        
        # CRITICAL: The 'finally' block with the delete call is INTENTIONALLY REMOVED
        # to allow the document state to persist across requests.
        # Cleanup is now handled by TTL and explicit session management APIs.
        
        self.graph_state.traversed_nodes.append("talk_to_doc")
        return state

    # --- Other methods (general_conversation, talk_to_data, etc.) remain unchanged ---
    def general_conversation(self, state: MessagesState):
        # This method remains unchanged
        state["current_node"] = "general_conversation"        
        try:
            default_system_prompt = "You are a helpful, respectful and honest assistant..."
            system_prompt = self.chat_request.systemPrompt if (self.chat_request.systemPrompt and self.chat_request.systemPrompt.strip()) else default_system_prompt
            system_message = SystemMessage(content=system_prompt + ". Note generate all your responses in markdown...")

            if self.model_provider == "open-ai":
                response = self.openai_chat.invoke([system_message] + state["messages"])
                state["messages"].append(response)
                self.chat_response.text_content = response.content
            elif self.model_provider == "bedrock":
                response = self.bedrock_model.invoke([system_message] + state["messages"])
                state["messages"].append(response)
                self.chat_response.text_content = response.content
            else:
                raise Exception(self.invalid_model_provide_msg)
        except Exception as ex:
            self.graph_state.failed_nodes.append("general_conversation")
            _logger.error(f"Exception has occurred while generate_file {ex}")
            raise ex
        self.graph_state.traversed_nodes.append("general_conversation")
        return state
