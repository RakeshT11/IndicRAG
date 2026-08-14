import sys
import os
import types
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_groq import ChatGroq
try:
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    import langchain_community.chat_models as _cm
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    class ChatVertexAI: pass
    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv
load_dotenv()
import logging
logging.basicConfig(level=logging.INFO)

class LoadModels:
    def __init__(self,
    embed_llm: str = "nvidia/nemotron-3-embed-1b",
    generator_llm: str = "llama-3.3-70b-versatile",
    eval_embeddings: str = "nvidia/nemotron-3-embed-1b",
    eval_llm: str = "openai/gpt-oss-120b",
    
    ):
        self.embed_llm=embed_llm
        self.generator_llm=generator_llm
        self.eval_embeddings=eval_embeddings
        self.eval_llm=eval_llm
        
    
    def load_embed_llm(self):
        """Load and return the  embedding model ."""
        
        nvidia_key = os.getenv("NVIDIA_API_KEY","").strip()
        if not nvidia_key:
            logging.warning(
                    "NVIDIA_API_KEY is not set."
                )
        try:
            embeddings = NVIDIAEmbeddings(
                    model=self.embed_llm,
                    chunk_size=300,
                )
            logging.info(f"Loaded NVIDIA embedding model: {self.embed_llm}")
            return embeddings
        except Exception as e:
            logging.error(f"Failed to load NVIDIAEmbeddings:{e}.")
    
    def load_generator_llm(self):
       
        try:
            
            logging.info(f"Initializing ChatGrok model: 'llama-3.3-70b-versatile' with temperature=0 and max_tokens=1024")
            llm = ChatGroq(
                    model_name=self.generator_llm,
                    temperature=0,
                    api_key=os.environ.get("GROQ_API_KEY"),
                    max_tokens=1024
                    )
            logging.info("ChatGrok model initialized successfully.")
            return llm
        except Exception as e:
            logging.error(f"Failed to initialize ChatGrok: {e}.")

    
    def load_eval_embeddings(self):

        eval_embeddings = LangchainEmbeddingsWrapper(
            NVIDIAEmbeddings(model=self.eval_embeddings)
            )
        return eval_embeddings
    
    def load_eval_llm(self):

        eval_llm = LangchainLLMWrapper(
            ChatNVIDIA(model=self.eval_llm,
                    max_tokens=4096,
            )
            )
        return eval_llm
