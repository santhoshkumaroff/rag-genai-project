from typing import TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class RAGQueryInput(BaseModel):
    query: str = Field(..., min_length=1, description="User query string")
    session_id: str = Field(default="default_session", description="Session identifier")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of results to retrieve")


class RAGQueryOutput(BaseModel):
    answer: str = Field(..., description="Generated answer from the RAG pipeline")
    sources: list[str] = Field(default_factory=list, description="Source document references")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")


class ToolResponse(BaseModel):
    tool_name: str = Field(..., description="Name of the tool that was executed")
    result: str = Field(..., description="Tool execution result")
    status: str = Field(default="success", description="Execution status")


class DocumentMetadata(BaseModel):
    source: str = Field(..., description="Source file path")
    page: int = Field(default=0, description="Page number")
    chunk_id: str = Field(default="", description="Chunk identifier")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Conversation messages"]
    context: str
