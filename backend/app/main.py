"""
FastAPI application for Tool-Enabled Agentic RAG.
Uses pure LangChain ReAct agent for intelligent question answering.
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from .models.schemas import QueryRequest, AgentResponse
from .agent.agent_executor import run_agent
from .agent.tools import vector_service
from .services.document_processor import process_document
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Tool-Enabled Agentic RAG API",
    description="Intelligent document Q&A with LangChain ReAct agent",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat", response_model=AgentResponse)
async def chat(request: QueryRequest):
    """
    Chat endpoint - processes user queries through the ReAct agent.
    
    The agent will:
    - Decide when to answer directly vs when to retrieve documents
    - Use retrieval as a tool, not a fixed pipeline
    - Perform multi-step reasoning when needed
    - Handle tool failures gracefully
    - Produce grounded, source-backed responses
    """
    print(f"--- POST /chat received: '{request.query[:50]}...' ---")
    
    # Run the agent
    response = run_agent(
        query=request.query,
        thread_id=request.thread_id or "default"
    )
    
    return response


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a document for RAG retrieval.
    Supports PDF and TXT files.
    """
    print(f"--- POST /upload received: '{file.filename}' ---")
    
    # Save the file temporarily
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process and index the document
        docs = process_document(file_path)
        vector_service.add_documents(docs)
        
        return {
            "message": f"Successfully indexed {file.filename}",
            "chunks": len(docs)
        }
        
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
