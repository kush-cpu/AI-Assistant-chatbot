from typing import Literal
import json

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.evaluation import compare_prompts, get_prompt_variants
from app.observability import (
    get_evaluation_events,
    get_evaluation_summaries,
    get_metrics_summary,
    get_observability_events,
)
from app.rag.pipeline import (
    chat_with_assistant,
    process_document,
    query_rag,
    stream_chat_with_assistant,
)

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    use_knowledge_base: bool = True


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class EvaluationRequest(BaseModel):
    question: str = Field(..., min_length=1)
    variants: list[str] | None = None


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _api_base_url(request: Request) -> str:
    return f"{_base_url(request)}/api/v1"


def _handle_error(exc: Exception, action: str) -> None:
    if isinstance(exc, EnvironmentError):
        raise HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=502, detail=str(exc))

    raise HTTPException(status_code=500, detail=f"{action} failed: {exc}")


def _model_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def _sse_event(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.get("/")
def root(request: Request):
    api_base_url = _api_base_url(request)
    return {
        "status": "running",
        "service": "AI Decision Assistant API",
        "api_base_url": api_base_url,
        "docs_url": f"{_base_url(request)}/docs",
        "endpoints": {
            "health": f"{api_base_url}/health",
            "capabilities": f"{api_base_url}/capabilities",
            "upload_document": f"{api_base_url}/documents/upload",
            "ask_document": f"{api_base_url}/rag/query",
            "chat": f"{api_base_url}/chat",
            "chat_stream": f"{api_base_url}/chat/stream",
            "observability": f"{api_base_url}/observability/events",
            "evaluation_metrics": f"{api_base_url}/evaluation/metrics",
            "prompt_comparison": f"{api_base_url}/evaluation/prompt-comparison",
        },
    }


@router.get("/api/v1/health")
def health():
    return {"status": "ok"}


@router.get("/api/v1/capabilities")
def capabilities(request: Request):
    api_base_url = _api_base_url(request)
    return {
        "api_base_url": api_base_url,
        "frontend_usage": {
            "base_url": api_base_url,
            "content_type_json": "application/json",
            "content_type_upload": "multipart/form-data",
        },
        "endpoints": [
            {
                "name": "Upload document",
                "method": "POST",
                "url": f"{api_base_url}/documents/upload",
                "body": {"file": "PDF file field in multipart/form-data"},
            },
            {
                "name": "Ask uploaded document",
                "method": "POST",
                "url": f"{api_base_url}/rag/query",
                "body": {"question": "What does the document say about ...?"},
            },
            {
                "name": "General assistant chat",
                "method": "POST",
                "url": f"{api_base_url}/chat",
                "body": {
                    "message": "User message",
                    "history": [
                        {"role": "user", "content": "Previous user message"},
                        {"role": "assistant", "content": "Previous assistant answer"},
                    ],
                    "use_knowledge_base": True,
                },
            },
            {
                "name": "Streaming assistant chat",
                "method": "POST",
                "url": f"{api_base_url}/chat/stream",
                "response": "text/event-stream",
                "events": [
                    {"type": "metadata", "sources": [], "used_knowledge_base": False},
                    {"type": "token", "content": "partial text"},
                    {"type": "done"},
                ],
            },
            {
                "name": "Observability events",
                "method": "GET",
                "url": f"{api_base_url}/observability/events",
                "query": {
                    "limit": 50,
                    "event_type": "rag_query | chat | chat_stream | document_upload",
                },
            },
            {
                "name": "Evaluation metrics",
                "method": "GET",
                "url": f"{api_base_url}/evaluation/metrics",
                "query": {"limit": 50},
            },
            {
                "name": "Prompt comparison",
                "method": "POST",
                "url": f"{api_base_url}/evaluation/prompt-comparison",
                "body": {
                    "question": "Question to evaluate against the uploaded document",
                    "variants": ["baseline", "concise", "risk_first"],
                },
            },
        ],
    }


@router.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        result = await process_document(file)
        return {
            "message": "Document processed",
            "document": {
                "filename": file.filename,
                "chunks": result["chunks"],
            },
        }
    except Exception as exc:
        _handle_error(exc, "Upload")


@router.post("/api/v1/rag/query")
async def rag_query(payload: QueryRequest):
    try:
        return await query_rag(payload.question)
    except Exception as exc:
        _handle_error(exc, "Query")


@router.post("/api/v1/chat")
async def chat(payload: ChatRequest):
    try:
        return await chat_with_assistant(
            message=payload.message,
            history=[_model_to_dict(item) for item in payload.history],
            use_knowledge_base=payload.use_knowledge_base,
        )
    except Exception as exc:
        _handle_error(exc, "Chat")


@router.post("/api/v1/chat/stream")
async def chat_stream(payload: ChatRequest):
    def event_generator():
        try:
            for event in stream_chat_with_assistant(
                message=payload.message,
                history=[_model_to_dict(item) for item in payload.history],
                use_knowledge_base=payload.use_knowledge_base,
            ):
                yield _sse_event(event)
        except Exception as exc:
            yield _sse_event({
                "type": "error",
                "detail": str(exc),
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/v1/observability/events")
async def observability_events(limit: int = 50, event_type: str | None = None):
    return {
        "events": get_observability_events(limit=limit, event_type=event_type),
    }


@router.get("/api/v1/evaluation/variants")
async def evaluation_variants():
    return {
        "variants": get_prompt_variants(),
    }


@router.get("/api/v1/evaluation/runs")
async def evaluation_runs(limit: int = 20, include_results: bool = False):
    runs = (
        get_evaluation_events(limit=limit)
        if include_results
        else get_evaluation_summaries(limit=limit)
    )
    return {
        "runs": runs,
    }


@router.get("/api/v1/evaluation/metrics")
async def evaluation_metrics(limit: int = 50):
    return get_metrics_summary(limit=limit)


@router.post("/api/v1/evaluation/prompt-comparison")
async def prompt_comparison(payload: EvaluationRequest):
    try:
        return compare_prompts(
            question=payload.question,
            variants=payload.variants,
        )
    except Exception as exc:
        _handle_error(exc, "Prompt comparison")


@router.post("/upload")
async def legacy_upload(file: UploadFile = File(...)):
    return await upload_document(file)


@router.post("/query")
async def legacy_query(q: str):
    return await rag_query(QueryRequest(question=q))
