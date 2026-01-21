import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def test_diagnostics():
    print("Step 1: Initializing Embeddings (HuggingFace)...")
    start = time.time()
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print(f"DONE: Embeddings initialized in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"FAILED: Embeddings initialization failed: {e}")
        return

    print("\nStep 2: Creating FAISS Index with dummy data...")
    start = time.time()
    try:
        texts = ["Hello world", "Agentic RAG is powerful"]
        vector_store = FAISS.from_texts(texts, embeddings)
        print(f"DONE: FAISS index created in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"FAILED: FAISS creation failed: {e}")
        return

    print("\nStep 3: Performing Similarity Search...")
    start = time.time()
    try:
        results = vector_store.similarity_search("How is RAG?", k=1)
        print(f"DONE: Search returned {len(results)} results in {time.time() - start:.2f}s")
        print(f"Result: {results[0].page_content}")
    except Exception as e:
        print(f"FAILED: Search failed: {e}")
        return

    print("\nStep 4: Testing persistence...")
    path = "./test_faiss"
    try:
        vector_store.save_local(path)
        print(f"DONE: Saved to {path}")
        new_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        print("DONE: Loaded back successfully")
    except Exception as e:
        print(f"FAILED: Persistence test failed: {e}")

if __name__ == "__main__":
    test_diagnostics()
