# Testing Guide: Multi-Step Agentic RAG

Follow these steps to verify that your agent is correctly planning, retrieving, and calculating.

## Prerequisites
1. **Ollama**: Ensure `ollama serve` is running and you have `llama3:latest`.
2. **Backend**: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
3. **Frontend**: `npm run dev`
4. **Knowledge**: Upload the [example.txt](file:///Users/techverito/ai/tool-enabled-agentic-rag/example.txt) file via the sidebar.

---

## Test Scenario 1: Basic Retrieval (The "Grounding" Test)
**Prompt:**  
> "What are the core hours I need to be online at TechFlow Inc?"

**Expected Agent Behavior:**
1. Calls **`retrieval_tool`** with query: `"core hours"`.
2. Finds "10:00 AM to 3:00 PM EST".
3. Returns a grounded answer with **example.txt** as the source.

---

## Test Scenario 2: Multi-Step Reasoning (Retrieval + Math)
**Prompt:**  
> "What is the monthly wellness stipend? If I save this stipend for 10 months to buy a $1,250 ergonomic desk, how much more money will I need?"

**Expected Agent Behavior:**
1. **Plan**: Realizes it needs to find the stipend amount first.
2. **Tool 1 (`retrieval_tool`)**: Finds the stipend is **$100**.
3. **Plan Update**: Now needs to calculate `100 * 10` and then `1,250 - result`.
4. **Tool 2 (`calculator_tool`)**: Performs the math ($1,000 savings).
5. **Tool 3 (`calculator_tool`)**: Subtracts from $1,250 ($250 needed).
6. **Final Answer**: "You get $100/month. Saving for 10 months gives you $1,000. You will need $250 more for the desk."

---

## Test Scenario 3: Safety & Fallback
**Prompt:**  
> "What is the policy for company cars?"

**Expected Agent Behavior:**
1. Calls **`retrieval_tool`**.
2. Finds no information.
3. Calls **`safety_fallback_tool`** or explains that the information is not in the handbook.
4. **Result**: No hallucination!

---

## Tips for Success
- **Watch the Terminal**: You can see the agent's "thinking" process (JSON logs) in the uvicorn terminal.
- **Source Badges**: Check the bottom of the chat bubble in the UI; it should show different icons for Documents vs. Tools.
- **Clear Session**: Use the "Clear Session" feature if the agent gets confused by previous unrelated questions.
