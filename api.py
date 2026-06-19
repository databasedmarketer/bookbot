from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import BookRAGPipeline

app = FastAPI(title="BookBot RAG API")


class ChatRequest(BaseModel):
    question: str
    language: str | None = "en"


class ChatResponse(BaseModel):
    answer: str
    processing_time: float
    sources: list


pipeline: BookRAGPipeline | None = None


@app.on_event("startup")
def startup_event():
    global pipeline
    pipeline = BookRAGPipeline()
    if not pipeline.initialize():
        raise RuntimeError("Failed to initialize RAG pipeline")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready")
    result = pipeline.query(req.question, language=req.language or "en")
    return ChatResponse(
        answer=result.get("answer", ""),
        processing_time=result.get("processing_time", 0.0),
        sources=result.get("retrieved_documents", []),
    )


