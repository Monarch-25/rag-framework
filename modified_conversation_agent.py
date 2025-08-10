from langgraph.graph import MessagesState
import os
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI
from langgraph.graph import MessagesState
from azure.identity import UsernamePasswordCredential
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.docstore.document import Document
import base64
import re
import markdown
import json
import boto3
from botocore.config import Config

from models.conversation import ChatRequest, ChatResponse, GraphState, FileData, CCConfig
from models.models import ChatLinkBase, TTDataMetaData
from common.logs_util import LogsUtil
from agents.rag_agent import RAGAgent
from utils import delete_documents_for_user
from chunking_strategies import get_chunker

_logger = LogsUtil("FID.AI fastapi -> Conversation Agent").get_logger()

class ConversationAgent:

    def __init__(self, chat_request: ChatRequest, chat_link_data: ChatLinkBase, chat_response: ChatResponse, graph_state: GraphState, user_files_content: str, is_file_data: bool, is_talk_to_data: bool, is_talk_to_doc: bool, ccconfig: CCConfig, chunking_strategy: str = "kamradt_modified"):
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

        if self.model_provider == "open-ai":
            self.openai_chat = AzureChatOpenAI(azure_deployment=self.chat_request.endPoint,api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),max_tokens=4096)
        elif self.model_provider == "bedrock":
            boto3_config = Config(
                connect_timeout=5,    
                read_timeout=600,     
                retries={'max_attempts': 3}
            )
            bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name="us-east-1",
                config=boto3_config
            )
            self.bedrock_model = ChatBedrock(model_id=self.chat_request.endPoint,client=bedrock_runtime,model_kwargs={"temperature": self.chat_request.temperature,"top_p": self.chat_request.top_p,"max_tokens": 4000},verbose=False)
        else:
            raise Exception("Invalid model provider")
        self.chat_response = chat_response
        self.graph_state = graph_state
        self.user_files_content = user_files_content
        self.is_file_data = is_file_data
        self.chat_link_data = chat_link_data
        self.is_talk_to_data = is_talk_to_data
        self.is_talk_to_doc = is_talk_to_doc
        self.ccconfig = ccconfig
        self.chunking_strategy = chunking_strategy 
        self.invalid_model_provide_msg = "Invalid model provider"


    def general_conversation(self, state: MessagesState):
        state["current_node"] = "general_conversation"        
        try:
            default_system_prompt = """You are a helpful, respectful and honest assistant...""" 
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

    def talk_to_doc(self, state: MessagesState):
        """
        Handles the 'talk to document' flow using a proper RAG pipeline.
        """
        state["current_node"] = "talk_to_doc"
        self.chat_response.persona = "talk_to_doc"
        
        # A unique ID for this user's session
        user_id = self.chat_link_data.session_id
        query = state["messages"][-1].content

        rag_agent = RAGAgent(chat_request=self.chat_request, chat_link_data=self.chat_link_data, ccconfig=self.ccconfig)
        
        try:
            # 1. Prepare documents
            docs = [Document(page_content=self.user_files_content, metadata={"source": "uploaded_document"})]
            
            # 2. Get the selected chunker and split documents
            text_splitter = get_chunker(self.chunking_strategy, rag_agent)
            chunks = text_splitter.split_documents(docs)
            _logger.info(f"Document split into {len(chunks)} chunks for user '{user_id}' using '{self.chunking_strategy}' strategy.")

            # 3. Add document chunks to the vector store for this user
            rag_agent.add_documents_for_user(user_id=user_id, documents=chunks)

            # 4. Get the specialized, secure retriever for the user
            retriever = rag_agent.get_retriever_for_user(user_id=user_id, user_docs=chunks)
            
            # 5. Invoke the retriever to get relevant context
            retrieved_docs = retriever.invoke(query)
            retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            _logger.info(f"Retrieved {len(retrieved_docs)} chunks for query.")

            # 6. Generate a response using the retrieved context
            system_prompt = f"""You are an AI assistant. Use the following retrieved context to answer the user's question. If the answer is not in the context, say so clearly. 
            
            Context:
            {retrieved_context}
            """
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
            _logger.error(f"Exception has occurred while talk_to_doc: {ex}")
            raise
        finally:
            # 8. CRITICAL: Clean up the user's data from the index
            _logger.info(f"Cleaning up data for user '{user_id}'.")
            delete_documents_for_user(
                raw_client=rag_agent.raw_client,
                index_name=rag_agent.index_name,
                user_id=user_id
            )
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
        state["messages"] = state["messages"] + [AIMessage(content = response["result"])]
        return state

    def filter_messages(self,messages: list):
        return messages[-1:]

    def identify_agent_and_ad_tools(self,state: MessagesState):
        prompt = ""
        if self.is_file_data is True:
            if self.is_talk_to_data is True:
                prompt = f""" \"agent\":\"talk_to_data\", \"tool\": \"xxx\" """
            else:
                prompt = f""" \"agent\":\"talk_to_doc\", \"tool\": \"xxx\" """
        else:
            prompt = f""" \"agent\":\\\"xxxx\\\",\"tool\": \"xxx\" """
        
        agent_tools = None
        if self.model_provider == "open-ai":
            agent_tools = self.openai_chat.invoke(input = prompt).content
        elif self.model_provider == "bedrock":    
            agent_tools = self.bedrock_model.invoke(input = prompt).content
        else:
            _logger.error(f"Failed to identify the agent and tool for {self.model_provider}")
            raise Exception(self.invalid_model_provide_msg)
        _logger.info(f"identify the agent and tool for {self.model_provider} :: {agent_tools}")
        state["agent_tools"] = agent_tools
        return state

    def generate_mermaid_diagram(self, state: MessagesState):
        self.chat_response.persona = "mermaid"
        prompt = f"""Create the diagram code with following: {state["messages"]}."""
        response = None
        if self.model_provider == "open-ai":
            response = self.openai_chat.invoke(input = prompt)
        elif self.model_provider == "bedrock":
            response = self.bedrock_model.invoke(input = prompt)
        else:
            raise Exception(self.invalid_model_provide_msg)
        return state

    def agent_identifier_condition(self, state):
        result = state.get("agent_tools")
        result = result.replace('```json', '').replace('```', '').strip()
        _logger.info(f"agent_identifier_condition called with agent_tools: {result}")
        try:
            data = json.loads(result)
            _logger.info(f"Parsed data: {data}")
            if data['agent'] == 'mermaid_agent':
                return "generate_mermaid_diagram"
            elif data['agent'] == 'talk_to_doc':
                return "talk_to_doc"
            elif data['agent'] == 'talk_to_data':
                return "talk_to_data"
            else:
                return "general_conversation"
        except Exception as e:
            _logger.error(f"Error in agent_identifier_condition: {e}")
            raise e

    def extract_mermaid_code(self, text):
        pattern = r'```mermaid\s*(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            raise Exception("Code does not have correct syntax for mermaid diagram")

    def generate_file(self, state: MessagesState):
        state["current_node"] = "generate_file"
        try:
            state["file_path"] = f"{os.environ.get('INPUT_FILE_PATH')}/{self.chat_link_data.session_id}.txt"
            if os.path.exists(state["file_path"]):
                os.remove(state["file_path"])
            response = state["messages"][-1]
            md_txt = markdown.markdown(response.content)
            with open(state["file_path"], 'w') as f:
                f.write(md_txt)
            if os.path.exists(state["file_path"]):
                with open(state["file_path"], "rb") as file:
                    file_content = file.read()
                encoded_content = base64.b64encode(file_content).decode('utf-8')
                file_data = FileData(data_base64=encoded_content,encoding="base64",filename=f"{self.chat_link_data.session_id}.txt")
                self.chat_response.files.append(file_data)
        except Exception as ex:
            self.graph_state.failed_nodes.append("generate_file")
            _logger.error(f"Exception has occurred while generate_file {ex}")
            raise ex
        self.graph_state.traversed_nodes.append("generate_file")
        return state
