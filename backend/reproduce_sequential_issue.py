import json
from langchain_core.messages import HumanMessage
from app.agent.graph import agent_app

def reproduce_issue():
    query = "Look up the daily trading hours in guide, then calculate the total for 5 days."
    print(f"--- Querying Agent: {query} ---")
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "thread_id": "test_sequential"
    }
    
    # We'll stream through the graph events
    for event in agent_app.stream(initial_state):
        for node_name, output in event.items():
            print(f"\n--- Node: {node_name} ---")
            if "messages" in output:
                for msg in output["messages"]:
                    print(f"Message Type: {type(msg).__name__}")
                    print(f"Content: {msg.content[:200]}...")
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"Tool Calls: {msg.tool_calls}")
            else:
                print(f"Output: {output}")

if __name__ == "__main__":
    reproduce_issue()
