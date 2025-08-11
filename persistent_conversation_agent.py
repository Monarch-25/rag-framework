import os
import boto3
import json
import re
import markdown
import base64
from botocore.config import Config
from azure.identity import UsernamePasswordCredential
from langgraph.graph import MessagesState
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.docstore.document import Document

# --- Project-specific Imports ---
from models.conversation import ChatRequest, ChatResponse, GraphState, FileData, CCConfig
from models.models import ChatLinkBase
from common.logs_util import LogsUtil
# Import the new stateful RAG agent
from doc_persistent_rag.persistent_rag_agent import PersistentRAGAgent
from chunking_strategies import get_chunker

_logger = LogsUtil("FID.AI fastapi -> PersistentConversationAgent").get_logger()

class PersistentConversationAgent:
    """
    Orchestrates a stateful, multi-turn conversation where document context
    persists throughout a user's session, and routes to various tools.
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

    def general_conversation(self, state: MessagesState):
        state["current_node"] = "general_conversation"        
        try:
            default_system_prompt = """You are a helpful, respectful and honest assistant..."""
            system_prompt = self.chat_request.systemPrompt if (self.chat_request.systemPrompt and self.chat_request.systemPrompt.strip()) else default_system_prompt
            system_message = SystemMessage(content=system_prompt + ". Note generate all your responses in markdown...")

            if self.model_provider == "open-ai":
                response = self.openai_chat.invoke([system_message] + state["messages"])
            elif self.model_provider == "bedrock":
                response = self.bedrock_model.invoke([system_message] + state["messages"])
            else:
                raise Exception(self.invalid_model_provide_msg)
            
            state["messages"].append(response)
            self.chat_response.text_content = response.content
        except Exception as ex:
            self.graph_state.failed_nodes.append("general_conversation")
            _logger.error(f"Exception has occurred while generate_file {ex}")
            raise ex
        self.graph_state.traversed_nodes.append("general_conversation")
        return state

    def talk_to_doc(self, state: MessagesState):
        """
        Handles the 'talk to document' flow using a persistent RAG pipeline.
        New documents are processed and added; duplicates are automatically skipped.
        """
        state["current_node"] = "talk_to_doc"
        self.chat_response.persona = "talk_to_doc"
        
        session_id = self.chat_link_data.session_id
        user_id = self.chat_link_data.user_id # Assuming user_id is available
        query = state["messages"][-1].content

        rag_agent = PersistentRAGAgent(chat_request=self.chat_request, chat_link_data=self.chat_link_data, ccconfig=self.ccconfig)
        
        try:
            # If new file content is provided, trigger the ingestion process.
            # The RAG agent's `add_documents_for_session` will handle de-duplication.
            if self.user_files_content:
                _logger.info(f"New file content received for session '{session_id}'. Processing for ingestion.")
                docs = [Document(page_content=self.user_files_content, metadata={"source": "uploaded_document"})]
                text_splitter = get_chunker(self.chunking_strategy, rag_agent)
                chunks = text_splitter.split_documents(docs)
                rag_agent.add_documents_for_session(user_id=user_id, session_id=session_id, documents=chunks)
            else:
                _logger.info(f"No new file content provided. Proceeding with existing documents in session '{session_id}'.")

            # The BM25 retriever needs some documents to be initialized.
            # We'll use the latest user content for this, but the main retrieval
            # will be against the full persistent session store.
            temp_docs_for_bm25 = [Document(page_content=self.user_files_content)] if self.user_files_content else []
            retriever = rag_agent.get_retriever_for_session(session_id=session_id, user_docs=temp_docs_for_bm25)
            
            retrieved_docs = retriever.invoke(query)
            retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            _logger.info(f"Retrieved {len(retrieved_docs)} chunks for query in session '{session_id}'.")

            system_prompt = f"""You are an AI assistant. Use the following retrieved context to answer the user's question... Context: {retrieved_context}"""
            system_message = SystemMessage(content=system_prompt)
            
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
        
        self.graph_state.traversed_nodes.append("talk_to_doc")
        return state

    def talk_to_data(self, state: MessagesState):
        state["current_node"] = "talk_to_data"
        self.chat_response.persona = "talk_to_data"
        from agents.db_agent import DBAgent
        dbAgent = DBAgent()
        response = dbAgent.talk_to_data(user_message=self.chat_request.userMessage,chat_link_data=self.chat_link_data,model_provider=self.model_provider,endpoint= self.chat_request.endPoint)
        self.chat_response.input_token += int(response["input_token"])
        self.chat_response.output_token += int(response["output_token"])
        self.chat_response.total_token += int(response["total_token"])
        self.chat_response.text_content = response["result"]
        state["messages"].append(AIMessage(content=response["result"]))
        return state

    def filter_messages(self, messages: list):
        return messages[-1:]

    def identify_agent_and_ad_tools(self, state: MessagesState):
        prompt = ""
        if self.is_file_data:
            prompt = f" \"agent\":\"talk_to_doc\", \"tool\": \"xxx\" "
        elif self.is_talk_to_data:
            prompt = f" \"agent\":\"talk_to_data\", \"tool\": \"xxx\" "
        else:
            prompt = f"You are the router... Agent Options [\"mermaid_agent\",\"general_agent\"]... Generate response in json like \"agent\":\"xxxx\",\"tool\": \"xxx\"."
        
        agent_tools = None
        if self.model_provider == "open-ai":
            agent_tools = self.openai_chat.invoke(input=prompt).content
        elif self.model_provider == "bedrock":    
            agent_tools = self.bedrock_model.invoke(input=prompt).content
        else:
            _logger.error(f"Failed to identify the agent and tool for {self.model_provider}")
            raise Exception(self.invalid_model_provide_msg)
        
        _logger.info(f"Identified agent and tool: {agent_tools}")
        state["agent_tools"] = agent_tools
        return state

    def agent_identifier_condition(self, state):
        result = state.get("agent_tools", '{}')
        result = result.replace('```json', '').replace('```', '').strip()
        _logger.info(f"agent_identifier_condition called with: {result}")
        try:
            data = json.loads(result)
            agent = data.get('agent', 'general_conversation')
            if agent == 'mermaid_agent':
                return "generate_mermaid_diagram"
            elif agent == 'talk_to_doc':
                return "talk_to_doc"
            elif agent == 'talk_to_data':
                return "talk_to_data"
            else:
                return "general_conversation"
        except json.JSONDecodeError:
            _logger.error(f"Failed to decode agent tools JSON: {result}")
            return "general_conversation"

    def generate_mermaid_diagram(self, state: MessagesState):
        self.chat_response.persona = "mermaid"
        prompt = f"Create the diagram code for the following: {state["messages"][-1].content}."
        response = None
        if self.model_provider == "open-ai":
            response = self.openai_chat.invoke(input=prompt)
        elif self.model_provider == "bedrock":
            response = self.bedrock_model.invoke(input=prompt)
        else:
            raise Exception(self.invalid_model_provide_msg)
        
        state["messages"].append(response)
        self.chat_response.text_content = response.content
        return state

    def extract_mermaid_code(self, text: str) -> str:
        pattern = r'```mermaid\s*(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        _logger.warning("Could not extract mermaid code from text.")
        return ""

    def generate_file(self, state: MessagesState):
        state["current_node"] = "generate_file"
        try:
            file_path = f"{os.environ.get('INPUT_FILE_PATH', '/tmp')}/{self.chat_link_data.session_id}.txt"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            response_content = state["messages"][-1].content
            md_txt = markdown.markdown(response_content)
            
            with open(file_path, 'w') as f:
                f.write(md_txt)
            
            with open(file_path, "rb") as file:
                file_content = file.read()
            
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            file_data = FileData(data_base64=encoded_content, encoding="base64", filename=f"{self.chat_link_data.session_id}.txt")
            self.chat_response.files.append(file_data)
        except Exception as ex:
            self.graph_state.failed_nodes.append("generate_file")
            _logger.error(f"Exception has occurred while generating file: {ex}")
            raise ex
        
        self.graph_state.traversed_nodes.append("generate_file")
        return state