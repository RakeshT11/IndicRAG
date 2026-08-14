import os
from typing import List
from datasets import load_dataset, Dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
logging.basicConfig(level=logging.INFO)
class Load_data:
    
    """Class responsible for loading, formatting, and chunking datasets for RAG."""
    
    def __init__(
        self,
        path: str = "BashitAli/Indian_history",
        split: str = "train",
        chunk_size: int = 300,
        chunk_overlap: int = 30
    ):
        self.path = path 
        self.split = split
        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap 
        self.dataset=self.load()        

    def load(self) -> Dataset:

        """Loading the dataset from Hugging Face Hub."""
        logging.info(f"Loading dataset '{self.path}' (split: '{self.split}')...")
        try:
            dataset = load_dataset(self.path, split=self.split,token=os.getenv("HF_TOKEN"))
            logging.info(f"Dataset loaded successfully with {len(dataset)} rows.")
        except Exception as e:
            logging.error(f"Failed to load dataset '{self.path}': {e}")
            raise e
        return dataset

    def process(self):
        """Converting raw dataset rows into structured LangChain Document objects."""
        dataset = self.dataset
        df = dataset.to_pandas()
        df["instruction"] = df["instruction"].astype(str).str.strip()
        df["response"] = df["response"].astype(str).str.strip()
        df_unique = df.drop_duplicates(subset=["instruction", "response"], keep="first")
        logging.info(f"Data set length after deduplication: {len(df_unique)}")
        documents: List[Document] = []
        for _,item in df_unique.iterrows():
            instruction = item['instruction']
            response = item['response']
            doc = Document(
                page_content=response,
                metadata={
                    "Question": instruction, 
                },
            )
            documents.append(doc)
        logging.info(f"Processed {len(documents)} document objects from dataset.")
        return documents

    def split_text(self) -> List[Document]:

        """Splitting processed documents into smaller chunks for vector embeddings."""
        documents = self.process()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        doc = text_splitter.split_documents(documents)
        logging.info(
            f"Split {len(documents)} documents into {len(doc)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})."
        )
        return doc


if __name__ == "__main__":
    data_loader = Load_data()
    chunks = data_loader.split_text()
    if chunks:
        print(f"Sample chunk (1 of {len(chunks)}):")
        print(chunks[0].page_content)
        print("Metadata:", chunks[0].metadata)
