from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List
from .models.schemas import QueryRequest, AgentResponse
from .agent.graph import agent_app
from .agent.tools import vector_service
from .services.document_processor import process_document
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Tool-Enabled Agentic RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Using the shared vector_service from agent.tools

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat", response_model=AgentResponse)
async def chat(request: QueryRequest):
    print(f"--- POST /chat received: '{request.query[:50]}' ---")
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "query": request.query,
        "thread_id": request.thread_id or "default"
    }
    
    # Run the agent
    result = agent_app.invoke(initial_state)
    
    return AgentResponse(
        answer=result.get("answer", "No answer generated."),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0)
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    print(f"--- POST /upload received: '{file.filename}' ---")
    # Save the file temporarily
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process and index the document
        docs = process_document(file_path)
        vector_service.add_documents(docs)
        return {"message": f"Successfully indexed {file.filename}", "chunks": len(docs)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
