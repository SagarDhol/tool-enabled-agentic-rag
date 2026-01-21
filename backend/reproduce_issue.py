import json
from langchain_core.messages import HumanMessage
from app.agent.graph import agent_app

def reproduce_issue():
    query = "What is the monthly wellness stipend? If I save this stipend for 10 months to buy a $1,250 ergonomic desk, how much more money will I need?"
    print(f"--- Querying Agent: {query} ---")
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "thread_id": "test_reproduction"
    }
    
    # We'll stream or just invoke and print snapshots
    # Since we want to see intermediate steps, we'll iterate through the graph events
    for event in agent_app.stream(initial_state):
        for node_name, output in event.items():
            print(f"\n--- Node: {node_name} ---")
            if "messages" in output:
                for msg in output["messages"]:
                    print(f"Message Type: {type(msg).__name__}")
                    print(f"Content: {msg.content}")
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"Tool Calls: {msg.tool_calls}")
            else:
                print(f"Output: {output}")

if __name__ == "__main__":
    reproduce_issue()
