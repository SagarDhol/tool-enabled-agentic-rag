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

def normalize_tool_call(tool_name: str, args: dict, full_data: dict = None, state: dict = None) -> dict:
    """
    Standardizes tool arguments to handle LLM variations and prevents ValidationErrors.
    """
    if not isinstance(args, dict):
        args = {}
    
    # 1. Tool-Specific Normalization
    if tool_name == "safety_fallback_tool":
        variants = ["request", "reason", "error", "message"]
        for variant in variants:
            if variant in args and "error_message" not in args:
                args["error_message"] = args.pop(variant)
                break
        if not args.get("error_message") and full_data:
            for variant in ["request", "reason", "error", "message", "error_message"]:
                if variant in full_data:
                    args["error_message"] = full_data[variant]
                    break
                    
    elif tool_name == "retrieval_tool":
        if "query" not in args:
            if "search_query" in args: args["query"] = args.pop("search_query")
            elif "q" in args: args["query"] = args.pop("q")
            elif full_data and "query" in full_data: args["query"] = full_data["query"]

    elif tool_name == "reasoning_tool":
        if "thought" not in args:
            if "analysis" in args: args["thought"] = args.pop("analysis")
            elif "reasoning" in args: args["thought"] = args.pop("reasoning")
            elif full_data and "thought" in full_data: args["thought"] = full_data["thought"]

    elif tool_name == "calculator_tool":
        if "expression" not in args:
            if "calculation" in args: args["expression"] = args.pop("calculation")
            elif "value" in args and "percentage" in args:
                val = str(args.pop("value")).replace("₹", "").replace(",", "").replace("$", "").strip()
                perc = str(args.pop("percentage")).replace("%", "").strip()
                args["expression"] = f"({perc}/100) * {val}"
            elif "capital" in args and "risk_percentage" in args:
                cap = str(args.pop("capital")).replace("₹", "").replace(",", "").replace("$", "").strip()
                risk = str(args.pop("risk_percentage")).replace("%", "").strip()
                args["expression"] = f"({risk}/100) * {cap}"
            elif "value" in args and "operation" in args and "factor" in args:
                val = str(args.pop("value")).replace("₹", "").replace(",", "").replace("$", "").strip()
                op = str(args.pop("operation")).lower()
                factor = str(args.pop("factor")).replace("%", "").strip()
                op_map = {"multiply": "*", "times": "*", "add": "+", "plus": "+", "subtract": "-", "minus": "-", "divide": "/"}
                symbol = op_map.get(op, "*")
                args["expression"] = f"{val} {symbol} {factor}"

    # 2. Generic fallback for single-argument tools
    tool_registry = {
        "calculator_tool": "expression",
        "retrieval_tool": "query",
        "reasoning_tool": "thought",
        "safety_fallback_tool": "error_message"
    }
    
    if tool_name in tool_registry:
        req_key = tool_registry[tool_name]
        if not args.get(req_key):
            # Check for generic placeholder keys from the prompt
            placeholders = ["query/thought/expression", "search_query", "calculation", "reasoning", "analysis"]
            for p in placeholders:
                if p in args:
                    args[req_key] = args.pop(p)
                    break
            
            if not args.get(req_key):
                if len(args) == 1:
                    key = list(args.keys())[0]
                    args[req_key] = args.pop(key)
                elif not args and full_data:
                    for variant in [req_key, "request", "reason", "analysis", "query", "thought"]:
                        if variant in full_data:
                            args[req_key] = full_data[variant]
                            break
            
            # Final Safe-Fail: Never pass empty args to a required tool parameter
            if not args.get(req_key):
                if state and state.get("messages"):
                    args[req_key] = state["messages"][-1].content[:100]
                else:
                    args[req_key] = "System-initiated tool call"

    return args

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

    # --- Tool Call Extraction and Normalization ---
    tool_calls_to_add = []
    
    # CASE 1: Model provided native tool calls
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            t_name = tc["name"]
            t_args = normalize_tool_call(t_name, tc["args"], state=state)
            tool_calls_to_add.append({"name": t_name, "args": t_args})

    # CASE 2: No native tool calls, try to extract from JSON text
    if not tool_calls_to_add:
        try:
            # Look for the last JSON block if multiple exist
            json_matches = re.findall(r'(\{.+\})', content, re.DOTALL)
            if json_matches:
                json_str = json_matches[-1]
                
                # Attempt to fix the most common error: {"args": {"something"}} -> {"args": {"query": "something"}}
                # If it looks like {"args": {"..."}}, replace it with {"args": {"query": "..."}}
                json_str = re.sub(r'"args":\s*\{\s*"([^"]+)"\s*\}', r'"args": {"query": "\1"}', json_str)
                
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    # Multi-stage fallback
                    data = {}
                    t_match = re.search(r'"tool":\s*"([^"]+)"', json_str)
                    if t_match: data["tool"] = t_match.group(1)
                    
                    a_match = re.search(r'"args":\s*(\{.*\})', json_str, re.DOTALL)
                    if a_match:
                        raw_args = a_match.group(1)
                        # Try to extract the VALUE after a colon (e.g., "query": "value")
                        val_match = re.search(r':\s*"([^"]+)"', raw_args)
                        if val_match: 
                            data["args"] = {"query": val_match.group(1)}
                        else:
                            # If no colon, maybe it's just {"value"}
                            val_match = re.search(r'"([^"]+)"', raw_args)
                            if val_match: data["args"] = {"query": val_match.group(1)}
                    else:
                        # Maybe args leaked to top level?
                        data["args"] = {}
                
                if "tool" in data:
                    t_name = data["tool"]
                    t_args = normalize_tool_call(t_name, data.get("args", {}), full_data=data, state=state)
                    tool_calls_to_add.append({"name": t_name, "args": t_args})
            
            elif "DONE" in content.upper():
                print("Planner signaled DONE")

        except Exception as e:
            print(f"Planner extraction error: {e}")

    # --- Duplication Check and Finalization ---
    final_tool_calls = []
    for tc in tool_calls_to_add:
        t_name = tc["name"]
        t_args = tc["args"]
        
        previous_tool_calls = [
            msg.tool_calls[0] for msg in state["messages"] 
            if hasattr(msg, "tool_calls") and msg.tool_calls
        ]
        is_duplicate = any(
            prev["name"] == t_name and prev["args"] == t_args 
            for prev in previous_tool_calls
        )
        
        if is_duplicate:
            print(f"Duplicate tool call detected: {t_name}. Signaling DONE.")
        else:
            print(f"\n--- Calling Tool: {t_name} ---")
            print(f"Tool call detected: {t_name} with normalized args: {t_args}")
            final_tool_calls.append({
                "name": t_name,
                "args": t_args,
                "id": f"call_{int(os.urandom(4).hex(), 16)}"
            })

    if final_tool_calls:
        response.tool_calls = final_tool_calls
    else:
        # If we have no tool calls (either DONE or all duplicates), clear them
        response.tool_calls = []
        if "DONE" in content.upper():
            print("Planner signaled DONE")
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
        print(f"\n--- Executing Tool: {tool_name} ---")
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
    
    content = response.content.strip()
    decision = "DONE" # Default
    
    try:
        # Attempt to parse JSON
        match = re.search(r'({.*})', content, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            decision = data.get("decision", "DONE").upper()
        else:
            # Fallback for plain text "DONE" or "CONTINUE"
            if "CONTINUE" in content.upper(): decision = "CONTINUE"
    except:
        if "CONTINUE" in content.upper(): decision = "CONTINUE"

    print(f"Evaluator decision: {decision}")
    
    # Check if we have gathered anything useful or hit a safety wall
    relevant_tool_messages = [
        msg for msg in state["messages"] 
        if isinstance(msg, ToolMessage)
    ]
    
    # We want to be sure we have ACTUAL document context or a failure, not just a planner guess
    has_factual_content = any(
        "Source:" in msg.content or "Providing safe fallback" in msg.content or "Reasoning recorded:" in msg.content or str(msg.content).replace(".", "").isdigit()
        for msg in relevant_tool_messages
    )
    
    # Loop prevention: Max 12 messages (approx 5-6 full loops)
    if len(state["messages"]) > 12:
        print("Hard loop prevention triggered: forcing DONE")
        return {"should_retry": False}

    # "Grumpy Evaluator" logic: 
    # Even if the LLM says DONE, if there is no factual content in the history, force CONTINUE
    if "DONE" in decision:
        if not has_factual_content and len(state["messages"]) < 5:
            print("Evaluator override: DONE signaled but no tools executed yet. Forcing CONTINUE.")
            return {"should_retry": True}
        return {"should_retry": False}
    elif decision == "CONTINUE":
        return {"should_retry": True}
    else:
        # Default fallback: if no factual content, continue
        return {"should_retry": not has_factual_content}

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
