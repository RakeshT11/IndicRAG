import os
from typing import Optional
from langchain_chroma import Chroma
from data_processing import Load_data
from dotenv import load_dotenv
load_dotenv()
import logging
from models import LoadModels
logging.basicConfig(level=logging.INFO)
class EmbedData:
    """Class to manage text embeddings and Chroma Vector Database initialization/persistence."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        persist_directory: str = "./chroma_db",
    ):
        self.persist_directory = os.getenv("PERSIST_DIRECTORY", persist_directory)
        self.embeddings = LoadModels().load_embed_llm()

    def build_vectordb(self, force_rebuild: bool = False) -> Chroma:

        """Building Chroma vector database from document splits and saves to persistent directory."""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory) and not force_rebuild:
            logging.info(f"Vector DB persistent directory '{self.persist_directory}' exists. Loading existing database...")
            return self.get_vectordb()

        logging.info("Building new Chroma vector database from document splits...")
        data_obj = Load_data()
        splits = data_obj.split_text()

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        logging.info(f"Created VectorDB successfully with {len(splits)} documents in '{self.persist_directory}'.")
        return vectordb
        
    def get_vectordb(self) -> Chroma:
        """Loads and returns an existing Chroma vector database from persistent storage."""
        if not os.path.exists(self.persist_directory):
            logging.warning(
                f"Persistent directory '{self.persist_directory}' does not exist. Building a new vector database..."
            )
            return self.build_vectordb()

        logging.info(f"Loading persistent VectorDB from '{self.persist_directory}'...")
        vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        return vectordb

        
if __name__ == "__main__":
    embedder = EmbedData()
    vdb = embedder.build_vectordb()
    logging.info(f"VectorDB initialization test complete. Persistence path: {embedder.persist_directory}")
    print(embedder.embeddings.embed_query("hi"))
