import os
from typing import Any, Dict, List, Optional,Annotated, TypedDict, Sequence
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
from embedding import EmbedData
# from langchain_core.runnables import  RunnablePassthrough
# from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph,START, END
from models import LoadModels
from dotenv import load_dotenv
import logging
load_dotenv()
logging.basicConfig(level=logging.INFO)

class AgentState(TypedDict):
    messages: List[BaseMessage]

class RAGPipeline:

    def __init__(
        self,
        k: int = 4,
        search_type: str = "similarity"        
    ):
        self.search_type=search_type
        self.k=k
        self.embedder =LoadModels().load_embed_llm()
        self.vectordb=EmbedData().get_vectordb()
        self.llm = LoadModels().load_generator_llm()
        self.retriever=self.vectordb.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.k}
            )      
        logging.info(f"Retriever configured with search_type='{self.search_type}' and k={self.k}")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context to answer the question. If the answer is not in the context, say so. Be concise."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}"),
        ])

    def _generate_response(self, state: AgentState) -> dict:
        """Generates a response using the LLM, context, and history."""

        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            query = last_message.content
        else:
            query = "Unknown"
        docs = self.retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        chain = (
            {
                "input": lambda x: x["messages"][-1].content,
                "chat_history": lambda x: x["messages"][:-1], 
                "context": lambda x: context,
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        response = chain.invoke({"messages": state["messages"], "context": context})
        return {"messages": [AIMessage(content=response)]}

    def invoke_query(self, query: str, chat_history: List[BaseMessage]) -> str:
       
        full_messages = chat_history + [HumanMessage(content=query)]
        initial_state = {"messages": full_messages}

        workflow = StateGraph(AgentState)
        workflow.add_node("generate", self._generate_response)
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", END)
        app = workflow.compile()
       
        try:
            result = app.invoke(initial_state)
            final_messages = result["messages"]
            return final_messages[-1].content
        except Exception as e:
            logger.error(f"Error invoking workflow: {e}")
            return f"Error processing query: {str(e)}"


