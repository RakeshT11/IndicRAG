import sys
import os
import ast
import pandas as pd
import streamlit as st
from datasets import Dataset
from dotenv import load_dotenv
from rag import RAGPipeline
from data_processing import Load_data
import types
try:
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    import langchain_community.chat_models as _cm
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    class ChatVertexAI: pass
    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
from ragas import evaluate
from ragas.metrics import faithfulness,answer_relevancy,context_recall,context_precision,answer_correctness
from ragas.run_config import RunConfig
from models import LoadModels
import logging
logging.basicConfig(level=logging.INFO)
load_dotenv()

class Evaluator:
    def __init__(self):
        self.rag = RAGPipeline(temperature=0.2)
        self.dataset = self.sample_dataset()
        models_obj=LoadModels()
        self.eval_embeddings=models_obj.load_eval_embeddings()
        self.eval_llm=models_obj.load_eval_llm()
        
    def sample_dataset(self):
        """Load a sample dataset for evaluation."""
        dataset = Load_data().load()
        dataset = dataset.shuffle(seed=42).select(range(20))
        dataset = dataset.rename_columns({
            "instruction": "question", 
            "response": "ground_truth" 
        })
        return dataset

    def rag_response(self, dataset):
        """Generate RAG responses and retrieved contexts."""
        responses = []
        for item in dataset:
            question = item["question"]
            gt = item["ground_truth"]
            gt_str = gt[0] if isinstance(gt, list) and len(gt) > 0 else str(gt)
            context_docs = self.rag.retriever.invoke(question)
            contexts = [doc.page_content for doc in context_docs]
            rag_answer = self.rag.chain.invoke(question)
            responses.append({
                "question": question,
                "ground_truth": gt_str,  
                "contexts": contexts,
                "answer": rag_answer
            })
        df_responses = pd.DataFrame(responses)
        df_responses.to_csv("eval_ds.csv", index=False)
        return df_responses

    def get_or_create_responses(self):
        """Retrieve existing RAG responses CSV or generate new ones."""
        if os.path.exists("eval_ds.csv"):
            df = pd.read_csv("eval_ds.csv")
        else:
            dataset = self.dataset 
            df = self.rag_response(dataset.to_pandas())
        def parse_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    res = ast.literal_eval(val)
                    if isinstance(res, list):
                        return res
                except Exception:
                    pass
                return [val]
            return []
        df["contexts"] = df["contexts"].apply(parse_list)
        def ensure_string(val):
            if isinstance(val, list):
                return val[0] if val else ""
            return str(val)
        df["ground_truth"] = df["ground_truths"].apply(ensure_string)
        del df['ground_truths']
        return Dataset.from_pandas(df)

    def evaluate_ragas(self):
        """Evaluate RAG pipeline using Ragas framework."""
        print("Starting Ragas Evaluation...")
        ragas_dataset = self.get_or_create_responses()
        print(f"Dataset columns: {ragas_dataset.column_names}")
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness
        ]

        config = RunConfig(
            timeout=300,         
            max_retries=10,
            max_wait=60      
        )
        results = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings,
            run_config=config 
        )
        df_results = results.to_pandas()
        df_results.to_excel("evaluation_results.xlsx", index=False)
        df_results.to_csv("evaluation_results.csv", index=False)
        st.title("Ragas Evaluation Results")
        st.write(df_results)
        print("Ragas evaluation completed.")
        return results

if __name__ == "__main__":
    ev = Evaluator()
    r = ev.evaluate_ragas()
    print(r)
