import os
import json
from app.agent.graph import agent_app
from langchain_core.messages import HumanMessage

def test_routing(query, expected_tool):
    print(f"--- Testing Query: '{query}' ---")
    print(f"--- Expected Tool: {expected_tool} ---")
    
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # We only care about the first step of the planner
    for event in agent_app.stream(initial_state):
        if "planner" in event:
            msg = event["planner"]["messages"][0]
            if msg.tool_calls:
                actual_tool = msg.tool_calls[0]["name"]
                print(f"--- Actual Tool: {actual_tool} ---")
                if actual_tool == expected_tool:
                    print("✅ PASSED")
                else:
                    print(f"❌ FAILED (Started with {actual_tool})")
            else:
                print(f"--- Actual: {msg.content if msg.content else 'DONE'} ---")
                if expected_tool == "DONE" or (not msg.tool_calls and expected_tool is None):
                    print("✅ PASSED")
                else:
                    print(f"❌ FAILED (No tool call)")
            break # Just check the first planner decision

if __name__ == "__main__":
    test_cases = [
        ("What is the share market?", "retrieval_tool"),
        ("Calculate 2% of ₹1,00,000.", "calculator_tool"),
        ("Why do institutional investors impact prices more?", "retrieval_tool"), # Should start with retrieval
        ("Tell me tomorrow's stock price.", "safety_fallback_tool"),
        ("Give me guaranteed stock tips.", "safety_fallback_tool"),
        ("What is SEBI tax rate for intraday trading?", "retrieval_tool"), # Should try to retrieve first
    ]
    
    for query, expected in test_cases:
        test_routing(query, expected)
        print("-" * 20)
