from app.agent.graph import planner_node
from langchain_core.messages import HumanMessage

def test_planner():
    print("--- Testing Planner Node ---")
    state = {
        "messages": [HumanMessage(content="Look up the daily trading hours in guide, then calculate the total for 5 days.")],
        "query": "Look up the daily trading hours in guide, then calculate the total for 5 days.",
        "thread_id": "test_planner"
    }
    
    # We call the planner_node directly
    result = planner_node(state)
    
    print("\n--- Result ---")
    if "messages" in result:
        msg = result["messages"][0]
        print(f"Content: {msg.content}")
        print(f"Tool Calls: {msg.tool_calls}")
    else:
        print(f"Output: {result}")

if __name__ == "__main__":
    test_planner()
