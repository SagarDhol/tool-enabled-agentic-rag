import os
import re
import json
from typing import List, Dict, Any, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from .state import AgentState
from .tools import tools, retrieval_tool, reasoning_tool, calculator_tool, safety_fallback_tool
from .prompts import SYSTEM_PROMPT, PLANNER_PROMPT, EVALUATOR_PROMPT
from ..models.schemas import Source, AgentResponse

# Initialize LLM
model = ChatOllama(model="llama3:latest", temperature=0)

def prepare_messages(state_messages: List[Any], system_message: SystemMessage) -> List[Any]:
    formatted = [system_message]
    tool_contents = []
    
    for msg in state_messages:
        if isinstance(msg, HumanMessage):
            # Keep only the FIRST human message for context (the user question)
            # or keep all if you want full chat history (safer for multi-turn)
            formatted.append(msg)
        elif isinstance(msg, ToolMessage):
            tool_contents.append(msg.content)
        elif isinstance(msg, AIMessage):
            # Keep AI messages that have content OR tool_calls
            if msg.content.strip() or msg.tool_calls:
                formatted.append(msg)
                
    # Add flattened tool output as a single HumanMessage at the end for fresh context
    if tool_contents:
        combined_tools = "\n\n".join(tool_contents)
        formatted.append(HumanMessage(content=f"LATEST UPDATED CONTEXT FROM DOCUMENTS:\n{combined_tools}"))
        
    return formatted

def planner_node(state: AgentState):
    """
    Decides the next course of action by parsing model output manually.
    """
    print("--- ENTERING PLANNER ---")
    messages = prepare_messages(state["messages"], SystemMessage(content=PLANNER_PROMPT))
    response = model.invoke(messages)
    
    # Attempt to parse JSON for manual tool calling
    import re
    content = response.content.strip()
    print(f"Planner raw output: {content[:100]}...")
    
    # Reset tool_calls to ensure we don't carry over old ones
    response.tool_calls = []

    try:
        # Improved regex to find the FIRST JSON object in the string
        # This handles preamble/postamble more effectively
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            # Try to fix common issues: unquoted keys or values
            # (Simple regex-based fix for common LLM mistakes)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback: simple text-based extraction if json.loads fails
                print("JSON.loads failed, attempting regex extraction fallback")
                data = {}
                tool_match = re.search(r'"tool":\s*"([^"]+)"', json_str)
                if tool_match:
                    data["tool"] = tool_match.group(1)
                args_match = re.search(r'"args":\s*(\{.*\})', json_str, re.DOTALL)
                if args_match:
                    try:
                        data["args"] = json.loads(args_match.group(1))
                    except:
                        data["args"] = {}
            
            if "tool" in data and "args" in data:
                tool_name = data["tool"]
                args = data["args"]
                
                # Proactive Loop Detection: Check if we've already tried this EXACT tool call
                previous_tool_calls = [
                    msg.tool_calls[0] for msg in state["messages"] 
                    if hasattr(msg, "tool_calls") and msg.tool_calls
                ]
                is_duplicate = any(
                    tc["name"] == tool_name and tc["args"] == args 
                    for tc in previous_tool_calls
                )
                
                if is_duplicate:
                    print(f"Duplicate tool call detected: {tool_name}. Signaling DONE.")
                    # If it's a duplicate, we signal DONE by not adding tool_calls
                else:
                    print(f"Tool call detected: {tool_name}")
                    response.tool_calls = [{
                        "name": tool_name,
                        "args": args,
                        "id": f"call_{int(os.urandom(4).hex(), 16)}"
                    }]
        elif "DONE" in content.upper():
            print("Planner signaled DONE")
            pass
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Planner parse error: {e}")
        pass
        
    return {"messages": [response]}

def tool_executor_node(state: AgentState):
    """
    Executes tool calls from the last message.
    """
    print("--- ENTERING TOOL EXECUTOR ---")
    last_message = state["messages"][-1]
    tool_outputs = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        print(f"Executing tool: {tool_name} with args: {args}")
        
        # Dispatch to the correct tool function
        if tool_name == "retrieval_tool":
            result = retrieval_tool.invoke(args)
        elif tool_name == "reasoning_tool":
            result = reasoning_tool.invoke(args)
        elif tool_name == "calculator_tool":
            result = calculator_tool.invoke(args)
        elif tool_name == "safety_fallback_tool":
            result = safety_fallback_tool.invoke(args)
        else:
            result = f"Error: Tool {tool_name} not found."
            
        tool_outputs.append(ToolMessage(
            tool_call_id=tool_call["id"],
            content=result
        ))
        
    return {"messages": tool_outputs}

def evaluator_node(state: AgentState):
    """
    Evaluates if the gathered information is enough to answer.
    """
    print("--- ENTERING EVALUATOR ---")
    messages = prepare_messages(state["messages"], SystemMessage(content=EVALUATOR_PROMPT))
    response = model.invoke(messages)
    
    # We expect the LLM to return a decision: "DONE" or "CONTINUE"
    content = response.content.strip().upper()
    print(f"Evaluator decision: {content}")
    
    # Check if we have gathered anything useful
    has_retrieved_info = any(
        isinstance(msg, ToolMessage) and "Source:" in msg.content 
        for msg in state["messages"]
    )
    
    if "DONE" in content:
        return {"should_retry": False}
    elif not has_retrieved_info and len(state["messages"]) < 5:
        # If we haven't even called a tool yet, continue
        return {"should_retry": True}
    else:
        # Safeguard: if we've already looped or have some info but LLM is unsure, force DONE
        print("Forcing end of retrieval loop to prevent infinite cycling.")
        return {"should_retry": False}

def final_answer_node(state: AgentState):
    """
    Generates the final grounded answer.
    """
    print("--- ENTERING FINAL ANSWER ---")
    messages = prepare_messages(state["messages"], SystemMessage(content=SYSTEM_PROMPT))
    response = model.invoke(messages)
    # ... (rest of final_answer_node logic remains same)
    
    content = response.content.strip()
    answer_text = content

    # Try to find a JSON-like structure {...}
    match = re.search(r'(\{.*\})', content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            data = json.loads(json_str)
            return {
                "answer": data.get("answer", answer_text),
                "sources": data.get("sources", []),
                "confidence": data.get("confidence", 0.0)
            }
        except json.JSONDecodeError:
            # Fallback for final answer if JSON is broken but "answer": "..." is present
            answer_match = re.search(r'"answer":\s*"([^"]+)"', json_str)
            if answer_match:
                return {
                    "answer": answer_match.group(1).replace('\\"', '"').replace('\\n', '\n'),
                    "sources": [],
                    "confidence": 0.5
                }
            pass

    return {
        "answer": answer_text,
        "sources": [],
        "confidence": 0.5
    }

def should_continue(state: AgentState) -> Literal["tools", "evaluator"]:
    """
    Router function to decide whether to execute tools or evaluate results.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "evaluator"

def check_evaluation(state: AgentState) -> Literal["planner", "final_answer"]:
    """
    Router function after evaluation.
    """
    if state.get("should_retry", False):
        return "planner"
    return "final_answer"

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("tools", tool_executor_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    should_continue,
    {
        "tools": "tools",
        "evaluator": "evaluator"
    }
)

workflow.add_edge("tools", "planner")

workflow.add_conditional_edges(
    "evaluator",
    check_evaluation,
    {
        "planner": "planner",
        "final_answer": "final_answer"
    }
)

workflow.add_edge("final_answer", END)

agent_app = workflow.compile()
