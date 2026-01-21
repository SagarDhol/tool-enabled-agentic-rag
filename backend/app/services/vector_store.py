import os
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

class VectorStoreService:
    def __init__(self, persist_directory: str = "./faiss_db"):
        print(f"--- Initializing VectorStoreService (persist: {persist_directory}) ---")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("--- Embeddings model loaded ---")
        self.persist_directory = persist_directory
        # For simplicity in this demo, we start with an empty FAISS index or load if exists
        if os.path.exists(self.persist_directory):
            print(f"--- Loading existing FAISS index from {self.persist_directory} ---")
            self.vector_store = FAISS.load_local(
                self.persist_directory, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("--- FAISS index loaded ---")
        else:
            print("--- Creating new FAISS index ---")
            # We initialize with a dummy document to create the index, then clear it
            # FAISS needs at least one document to initialize
            self.vector_store = FAISS.from_texts(
                ["initialization text"], 
                self.embeddings,
                metadatas=[{"source": "init"}]
            )
            print("--- FAISS index initialized ---")

    def add_documents(self, documents: List[Document]):
        print(f"--- Adding {len(documents)} documents to FAISS ---")
        self.vector_store.add_documents(documents)
        print("--- Saving FAISS index locally ---")
        self.vector_store.save_local(self.persist_directory)
        print("--- FAISS index saved ---")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        print(f"--- Searching FAISS for: '{query}' ---")
        results = self.vector_store.similarity_search(query, k=k)
        print(f"--- Found {len(results)} results ---")
        return results

    def as_retriever(self, k: int = 5):
        return self.vector_store.as_retriever(search_kwargs={"k": k})
