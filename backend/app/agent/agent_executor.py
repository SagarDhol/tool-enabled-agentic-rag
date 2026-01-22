"""
Pure RAG Agent - ALWAYS retrieves from documents first.
Only answers based on uploaded documents, not general knowledge.

Key behavior:
- Always calls retrieval_tool first for any question
- Only uses knowledge from uploaded documents
- Says "I don't have this information" if not in documents
- Uses calculator only for math on retrieved data
"""
import os
import re
import json
from typing import Optional, List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .tools import retrieval_tool, reasoning_tool, calculator_tool
from ..models.schemas import Source, AgentResponse
from dotenv import load_dotenv

load_dotenv()

# Constants
MAX_ITERATIONS = 10

# Flexible ReAct prompt - Strong enforcement of tool usage
REACT_SYSTEM_PROMPT = """You are a precise AI assistant that MUST use tools to answer questions accurately and safely.

## Available Tools

1. **retrieval_tool(query)**: Search uploaded documents for specific information.
   - Use this for ANY question about React, company policies, manuals, or specific guides.
   - Example: "What is useState according to the doc?", "What are the types of components?"

2. **calculator_tool(expression)**: Perform mathematical calculations.
   - Use this for ALL math, including simple arithmetic. Do NOT do mental math.
   - Example: "4 + 3", "9 - 3", "12% of 50,000".

3. **reasoning_tool(thought)**: Document deep analysis or logical steps.
   - Use this before answering complex "Why" or "If" questions.
   - Example: "Why are functional components recommended?", "What happens if policy is violated?"

## Decision Rules - MANDATORY

### 1. Simple Greetings ONLY (Direct Answer)
- You may only answer directly (without tools) for simple greetings like "Hi", "Hello", or "Who are you?".

### 2. Document Questions (Mandatory Retrieval)
- If the question is about React, the Virtual DOM, state, props, or any technical topic mentioned in the guides, you MUST use `retrieval_tool` first.
- Do NOT answer from your internal knowledge even if you know the answer.

### 3. Math (Mandatory Calculator)
- For ANY numerical calculation, you MUST use `calculator_tool`.

### 4. Complex Analysis (Mandatory Reasoning)
- For "Why?", "Compare", or "If..." questions, use `reasoning_tool` to think, then likely `retrieval_tool` to check facts, then provide the final answer.

## Response Format

To use a tool, respond ONLY with:
```json
{"action": "tool_name", "action_input": "input value"}
```

When you have the final answer based on tool results:
```json
{"action": "final_answer", "action_input": "Your complete answer here"}
```

## Examples

Q: "What is the role of the Virtual DOM?"
→ {"action": "retrieval_tool", "action_input": "Virtual DOM role function"}

Q: "4 + 3"
→ {"action": "calculator_tool", "action_input": "4 + 3"}

Q: "Why should hooks be called at the top level?"
→ {"action": "reasoning_tool", "action_input": "Analyzing hook call rules and component lifecycle..."}

REMEMBER: If you answer a technical or mathematical question without a tool call, you have FAILED your objective."""


def get_llm(model_name: str = "llama3:latest", temperature: float = 0):
    """Get the Ollama LLM instance."""
    return ChatOllama(model=model_name, temperature=temperature)


def parse_llm_action(content: str) -> Dict[str, Any]:
    """Parse LLM response to extract action and input."""
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'(\{[^{}]*"action"[^{}]*\})',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                # Clean any trailing non-JSON chars that might have been caught
                text = match.group(1).strip()
                if not text.endswith('}'):
                    text = text[:text.rfind('}')+1]
                data = json.loads(text)
                if "action" in data:
                    return data
            except json.JSONDecodeError:
                continue
    
    # If no JSON found, check if it looks like a direct answer that SHOULD have been a tool call
    content_lower = content.lower()
    technical_keywords = ['react', 'hook', 'state', 'props', 'dom', 'component', 'useeffect', 'usestate', 'policy', 'guide']
    math_patterns = [r'\d+\s*[+\-*/]\s*\d+', r'calculate', r'percentage', r'sum', r'total']
    
    is_technical = any(kw in content_lower for kw in technical_keywords)
    is_math = any(re.search(pat, content_lower) for pat in math_patterns)
    
    if (is_technical or is_math) and len(content) > 10:
        return {"action": "require_tool", "action_input": content}
        
    return {"action": "final_answer", "action_input": content}


def execute_tool(action: str, action_input: str) -> str:
    """Execute the specified tool and return result."""
    # Special internal action
    if action == "require_tool":
        return "ERROR: You provided a direct answer but a tool call (JSON) was required for this technical/mathematical question. Please use retrieval_tool or calculator_tool now."

    print(f"--- Executing tool: {action}({action_input}) ---")
    
    tool_map = {
        "retrieval_tool": retrieval_tool,
        "reasoning_tool": reasoning_tool,
        "calculator_tool": calculator_tool,
    }
    
    if action not in tool_map:
        return f"Unknown tool: {action}. Available tools: retrieval_tool, calculator_tool, reasoning_tool"
    
    try:
        if action == "retrieval_tool":
            result = tool_map[action].invoke({"query": action_input})
        elif action == "calculator_tool":
            result = tool_map[action].invoke({"expression": action_input})
        elif action == "reasoning_tool":
            result = tool_map[action].invoke({"thought": action_input})
        else:
            result = tool_map[action].invoke(action_input)
        return str(result)
    except Exception as e:
        return f"Tool error: {str(e)}"


def run_agent_loop(
    query: str,
    llm,
    chat_history: Optional[List] = None,
    max_iterations: int = MAX_ITERATIONS
) -> Dict[str, Any]:
    """
    Run the ReAct agent loop with strict tool enforcement.
    """
    messages = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        *(chat_history or []),
        HumanMessage(content=f"User question: {query}\n\nREMEMBER: You must use JSON format to call a tool if this is technical or mathematical. Do not answer directly unless it is a simple greeting.")
    ]
    
    tools_used = []
    
    for iteration in range(max_iterations):
        print(f"\n{'='*20} ITERATION {iteration + 1}/{max_iterations} {'='*20}")
        print(f"--- [Thinking] Agent is analyzing information... ---")
        
        response = llm.invoke(messages)
        content = response.content.strip()
        
        parsed = parse_llm_action(content)
        action = parsed.get("action", "final_answer")
        action_input = parsed.get("action_input", content)
        
        # Check if we are forcing a tool call
        if action == "require_tool" and iteration == 0:
            print(f"--- [Warning] Agent tried to answer directly. Forcing tool usage... ---")
            messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content="You answered directly but this requires a tool. Please use the appropriate tool (JSON format) to verify this information from the documents or perform the calculation."))
            continue

        if action == "final_answer":
            print(f"--- [Complete] Agent has found the final answer. ---")
            print(f"--- Tools used in this run: {tools_used if tools_used else 'None'} ---")
            return {"output": action_input, "messages": messages, "tools_used": tools_used}
        
        # Execute the tool
        print(f"--- [Tool Starting] Tool: {action} | Input: {action_input[:100]}... ---")
        
        # Check for internal error action
        if action == "require_tool":
            tool_result = execute_tool(action, action_input)
        else:
            tools_used.append(action)
            tool_result = execute_tool(action, action_input)
            
        print(f"--- [Tool Finished] {action} execution complete. ---")
        
        messages.append(AIMessage(content=content))
        
        # Provide context for next step
        messages.append(HumanMessage(content=f"""Tool Result from {action}:
{tool_result}

Now decide your next step based on this result. Provide either another tool action or a final_answer (in JSON format)."""))
    
    print(f"--- [Warning] Max iterations reached. ---")
    final_response = llm.invoke(messages + [
        HumanMessage(content="Provide your final answer now.")
    ])
    return {"output": final_response.content, "messages": messages, "tools_used": tools_used}


def parse_agent_response(output: Any) -> Dict[str, Any]:
    """Parse agent output into structured response format."""
    # Ensure output is a string
    output_str = str(output)
    
    # Try to extract JSON
    json_match = re.search(r'\{.*\}', output_str, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "answer" in data or "action_input" in data:
                # If JSON contains sources, it's likely document-based
                has_sources_in_json = bool(data.get("sources"))
                return {
                    "answer": data.get("answer", data.get("action_input", output_str)),
                    "sources": data.get("sources", []),
                    "confidence": data.get("confidence", 0.85)
                }
        except json.JSONDecodeError:
            pass
    
    # Extract sources from text
    sources = []
    source_patterns = [
        r'Source:\s*([^\s\)\n]+)',
        r'\(from\s+([^\s\)]+)\)',
        r'\(Source:\s*([^\)]+)\)',
        r'from\s+([^\s,\.]+\.txt)',
        r'-- Document \d+ \(Source: ([^\)]+)\)',
    ]
    for pattern in source_patterns:
        for match in re.findall(pattern, output_str, re.IGNORECASE):
            clean = match.strip('(),.: ')
            if clean and clean not in [s.get("name", "") for s in sources]:
                sources.append({"type": "document", "name": clean, "reference": clean})
    
    return {
        "answer": output_str,
        "sources": sources,
        "confidence": 0.9 if sources else 0.75
    }


# Singleton LLM
_llm = None

def get_agent_llm():
    """Get or create singleton LLM."""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def run_agent(
    query: str,
    thread_id: str = "default",
    chat_history: Optional[List] = None
) -> AgentResponse:
    """
    Run the agent with the given query.
    
    The agent will:
    - Use tools conditionally (skip if not needed)
    - Perform multi-step reasoning/chaining if required
    - Support document retrieval, calculations, and complex reasoning
    """
    print(f"\n{'='*60}")
    print(f"--- New Query: '{query}' ---")
    print(f"{'='*60}")
    
    llm = get_agent_llm()
    
    try:
        result = run_agent_loop(query=query, llm=llm, chat_history=chat_history)
        output = result.get("output", "I couldn't generate a response.")
        tools_used = result.get("tools_used", [])
        parsed = parse_agent_response(output)
        
        print(f"--- Agent completed successfully ---")
        print(f"--- Tools used: {tools_used} ---")
        
        return AgentResponse(
            answer=parsed["answer"],
            sources=[Source(**s) if isinstance(s, dict) else s for s in parsed["sources"]],
            confidence=parsed["confidence"]
        )
        
    except Exception as e:
        error_msg = f"Agent error: {str(e)}"
        print(f"--- {error_msg} ---")
        
        return AgentResponse(
            answer=f"I encountered an error: {str(e)}. Please try rephrasing your question.",
            sources=[],
            confidence=0.0
        )
