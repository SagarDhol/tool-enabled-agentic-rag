from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Source(BaseModel):
    type: str # Usually 'document' or 'tool'
    name: str
    reference: str

class AgentResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)
    confidence: float

class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

class DocumentMetadata(BaseModel):
    filename: str
    doc_id: str
    page: Optional[int] = None
