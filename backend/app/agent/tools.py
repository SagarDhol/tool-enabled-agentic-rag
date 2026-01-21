from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from ..services.vector_store import VectorStoreService
from ..models.schemas import Source
import json

# Initialize VectorStoreService
# In a real app, this should be injected or handled via a singleton
vector_service = VectorStoreService()

@tool
def retrieval_tool(query: str) -> str:
    """
    Retrieves relevant document chunks based on the query.
    """
    print(f"--- Calling retrieval_tool with query: '{query}' ---")
    docs = vector_service.similarity_search(query, k=3)
    if not docs:
        print("--- No docs found in retrieval_tool ---")
        return "No relevant context found."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"--- Document {i+1} (Source: {doc.metadata.get('source', 'unknown')}) ---\n{doc.page_content}")
    
    final_output = "\n\n".join(results)
    print(f"--- retrieval_tool returning {len(results)} results ---")
    return final_output

@tool
def reasoning_tool(thought: str) -> str:
    """
    Use this tool to record complex reasoning steps, decompose problems, 
    or perform step-by-step analysis. This tool doesn't call external APIs 
    but helps in structuring the agent's internal logic.
    """
    # This tool is mostly used for the agent to "talk to itself" or document its reasoning
    return f"Reasoning recorded: {thought}"

@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluates a mathematical expression.
    """
    try:
        # NOTE: eval is dangerous in production, using a safer alternative or 
        # restricted environment is recommended. For this example, we'll use a simple eval.
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def safety_fallback_tool(error_message: str) -> str:
    """
    Handles tool failures or empty results gracefully.
    """
    return f"Providing safe fallback due to: {error_message}. I cannot provide a reliable answer at this time."

# List of tools to be used by the agent
tools = [retrieval_tool, reasoning_tool, calculator_tool, safety_fallback_tool]
