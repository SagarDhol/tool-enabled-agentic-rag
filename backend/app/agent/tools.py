"""
Tools available to the LangChain ReAct agent.
Each tool includes robust error handling and clear descriptions.
"""
from typing import List
from langchain_core.tools import tool
from ..services.vector_store import VectorStoreService

# Initialize VectorStoreService singleton
_vector_service = None

def get_vector_service() -> VectorStoreService:
    """Get or create the singleton vector store service."""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorStoreService()
    return _vector_service

# Export for backward compatibility
vector_service = get_vector_service()


@tool
def retrieval_tool(query: str) -> str:
    """
    Search uploaded documents for relevant information.
    
    Use this tool when you need to find information from the user's documents.
    Provide a clear, specific search query for best results.
    
    Args:
        query: A clear search query describing what information you're looking for.
    
    Returns:
        Relevant document chunks with source information, or a message if nothing found.
    """
    print(f"\n[TOOL START] retrieval_tool | Query: '{query}'")
    
    try:
        vs = get_vector_service()
        print(f"[VERBOSE] retrieval_tool: Initialized Vector Service. Starting search...")
        docs = vs.similarity_search(query, k=3)
        
        if not docs:
            print(f"[TOOL FINISH] retrieval_tool | Status: No results found.")
            return "No relevant documents found for this query. The user may need to upload relevant documents first."
        
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            if '/' in source:
                source = source.split('/')[-1]
            results.append(
                f"--- Document {i+1} (Source: {source}) ---\n{doc.page_content}"
            )
        
        output = "\n\n".join(results)
        print(f"[TOOL FINISH] retrieval_tool | Status: Successfully found {len(results)} chunks.")
        return output
        
    except Exception as e:
        error_msg = f"Error searching documents: {str(e)}"
        print(f"[TOOL ERROR] retrieval_tool | {error_msg}")
        return f"Tool error: {error_msg}. Please try a different query or check if documents are uploaded."


@tool
def reasoning_tool(thought: str) -> str:
    """
    Record a reasoning step or analysis for complex problems.
    
    Use this tool to document your thinking process when:
    - Breaking down complex questions into steps
    - Synthesizing information from multiple sources
    - Explaining your analytical approach
    
    Args:
        thought: Your reasoning step, analysis, or synthesis to record.
    
    Returns:
        Confirmation that the reasoning was recorded.
    """
    print(f"\n[TOOL START] reasoning_tool | Reasoning: '{thought[:100]}...'")
    print(f"[VERBOSE] reasoning_tool: Documenting analysis step...")
    print(f"[TOOL FINISH] reasoning_tool | Status: Step recorded.")
    return f"Reasoning recorded: {thought}"


@tool
def calculator_tool(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    
    Use this tool for calculations like:
    - Percentages: "0.02 * 100000" for 2% of ₹1,00,000
    - Arithmetic: "45 + 23 * 2"
    - Financial math: "(1500000 * 0.05) / 12" for monthly interest
    
    Args:
        expression: A mathematical expression using +, -, *, /, (), and numbers only.
    
    Returns:
        The numerical result or an error message.
    """
    print(f"\n[TOOL START] calculator_tool | Expression: '{expression}'")
    
    try:
        clean_expr = expression
        for char in ['₹', '$', '€', ',', ' ']:
            clean_expr = clean_expr.replace(char, '')
        
        print(f"[VERBOSE] calculator_tool: Cleaned expression to: {clean_expr}")
        
        allowed_chars = set('0123456789+-*/.()% ')
        if not all(c in allowed_chars for c in clean_expr):
            invalid_chars = [c for c in clean_expr if c not in allowed_chars]
            print(f"[TOOL ERROR] calculator_tool | Invalid characters: {invalid_chars}")
            return f"Error: Expression contains invalid characters: {invalid_chars}. Use only numbers and operators (+, -, *, /, %, ())"
        
        clean_expr = clean_expr.replace('%', '/100')
        
        result = eval(clean_expr, {"__builtins__": {}}, {})
        
        if isinstance(result, float):
            if result == int(result):
                result = int(result)
            else:
                result = round(result, 2)
        
        print(f"[TOOL FINISH] calculator_tool | Status: Calculation complete. Result: {result}")
        return str(result)
        
    except ZeroDivisionError:
        print(f"[TOOL ERROR] calculator_tool | Division by zero attempted.")
        return "Error: Cannot divide by zero."
    except SyntaxError:
        print(f"[TOOL ERROR] calculator_tool | Syntax error in expression.")
        return f"Error: Invalid expression syntax. Please use a valid mathematical expression."
    except Exception as e:
        print(f"[TOOL ERROR] calculator_tool | {str(e)}")
        return f"Error evaluating expression: {str(e)}"


# List of all tools for the agent
tools = [retrieval_tool, reasoning_tool, calculator_tool]
