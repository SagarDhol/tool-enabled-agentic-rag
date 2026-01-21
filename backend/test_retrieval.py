import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def test_retrieval():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "./faiss_db"
    
    if not os.path.exists(persist_directory):
        print(f"Error: {persist_directory} does not exist.")
        return

    vector_store = FAISS.load_local(
        persist_directory, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    queries = [
        "What are the core hours?",
        "TechFlow core hours",
        "When should I be online at TechFlow?",
        "vacation days"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = vector_store.similarity_search(query, k=3)
        for i, res in enumerate(results):
            print(f"Result {i+1}:")
            print(f"Content: {res.page_content}")
            print(f"Metadata: {res.metadata}")

if __name__ == "__main__":
    test_retrieval()
