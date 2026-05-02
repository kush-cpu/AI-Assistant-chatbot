import os
from pathlib import Path
from urllib.parse import quote
from urllib import request
from urllib.error import HTTPError, URLError
from dotenv import load_dotenv
from fastapi import UploadFile
from app.rag.processor import extract_text_from_pdf, chunk_text
from app.storage.vector_store import VectorStore
from app.rag.prompt import build_prompt
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
RETRIEVAL_CANDIDATES = 8
CONTEXT_CHUNKS = 5
MAX_CONTEXT_CHARS = 700
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
        "key_insights": {"type": "ARRAY", "items": {"type": "STRING"}},
        "risks": {"type": "ARRAY", "items": {"type": "STRING"}},
        "recommendation": {"type": "STRING"},
        "sources": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "page": {"type": "INTEGER"},
                    "snippet": {"type": "STRING"},
                },
                "required": ["page", "snippet"],
            },
        },
    },
    "required": ["answer", "key_insights", "risks", "recommendation", "sources"],
}
CHAT_SYSTEM_PROMPT = """
You are a helpful AI assistant. Answer clearly and directly. If the user asks
about uploaded knowledge-base content, use the provided document context when it
is available. If no relevant context is available, answer as a general assistant.
"""

vector_store = VectorStore()
vector_store.load()

UPLOAD_DIR = "data/uploads/"


def _get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
    return api_key


def _post_gemini(path: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{GEMINI_API_URL}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": _get_gemini_api_key(),
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed ({exc.code}): {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Gemini API is not reachable: {exc}") from exc


def _iter_gemini_sse(path: str, payload: dict, timeout: int = 180):
    data = json.dumps(payload).encode("utf-8")
    separator = "&" if "?" in path else "?"
    req = request.Request(
        f"{GEMINI_API_URL}{path}{separator}alt=sse",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "x-goog-api-key": _get_gemini_api_key(),
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue

                data_line = line.removeprefix("data:").strip()
                if data_line == "[DONE]":
                    break

                try:
                    yield json.loads(data_line)
                except json.JSONDecodeError:
                    continue
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini streaming request failed ({exc.code}): {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Gemini streaming API is not reachable: {exc}") from exc


def _effective_gemini_model() -> str:
    return "gemini-2.5-flash" if GEMINI_MODEL == "gemini-1.5-flash" else GEMINI_MODEL


def _extract_response_text(response: dict) -> str:
    candidates = response.get("candidates", [])
    if not candidates:
        return ""

    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def call_gemini_llm(
    prompt: str,
    *,
    response_mime_type: str = "text/plain",
    response_schema: dict | None = None,
    max_output_tokens: int = 1200,
) -> str:
    model = quote(f"models/{_effective_gemini_model()}", safe="/")
    generation_config = {
        "temperature": 0.2,
        "topP": 0.9,
        "maxOutputTokens": max_output_tokens,
        "responseMimeType": response_mime_type,
    }

    if response_schema:
        generation_config["responseSchema"] = response_schema

    response = _post_gemini(
        f"/{model}:generateContent",
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": generation_config,
        },
        timeout=180,
    )

    return _extract_response_text(response)


def call_gemini_llm_stream(prompt: str, max_output_tokens: int = 1200):
    model = quote(f"models/{_effective_gemini_model()}", safe="/")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "text/plain",
        },
    }

    for event in _iter_gemini_sse(f"/{model}:streamGenerateContent", payload):
        text = _extract_response_text(event)
        if text:
            yield text


def _embed_text(text: str, task_type: str) -> list[float]:
    model = quote(f"models/{GEMINI_EMBEDDING_MODEL}", safe="/")
    response = _post_gemini(
        f"/{model}:embedContent",
        {
            "content": {
                "parts": [{"text": text}],
            },
            "taskType": task_type,
        },
        timeout=120,
    )
    return response["embedding"]["values"]


def embed_texts_gemini(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    embeddings = []

    for text in texts:
        embeddings.append(_embed_text(text, task_type))

    return embeddings


def _ensure_vector_store_dim(dim: int) -> None:
    global vector_store

    if vector_store.index.d != dim:
        vector_store = VectorStore(dim=dim)


def _reset_vector_store(dim: int) -> None:
    global vector_store

    vector_store = VectorStore(dim=dim)


def _rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    query_terms = {
        term.lower()
        for term in query.split()
        if len(term) > 2
    }

    def score(chunk: dict) -> int:
        text = chunk.get("text", "").lower()
        return sum(1 for term in query_terms if term in text)

    return sorted(
        enumerate(chunks),
        key=lambda item: (score(item[1]), -item[0]),
        reverse=True,
    ) # type: ignore


def _limit_chunks(
    chunks: list[dict],
    limit: int = CONTEXT_CHUNKS,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> list[dict]:
    limited_chunks = []

    for chunk in chunks[:limit]:
        limited_chunk = dict(chunk)
        limited_chunk["text"] = limited_chunk.get("text", "")[:max_chars]
        limited_chunks.append(limited_chunk)

    return limited_chunks


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen = set()
    unique_chunks = []

    for chunk in chunks:
        text = " ".join(chunk.get("text", "").lower().split())
        key = (chunk.get("page"), text[:220])
        if key in seen:
            continue

        seen.add(key)
        unique_chunks.append(chunk)

    return unique_chunks


def _dedupe_source_items(sources: list[dict]) -> list[dict]:
    seen = set()
    unique_sources = []

    for source in sources:
        snippet = str(source.get("snippet") or source.get("text") or "").strip()
        key = (source.get("page"), " ".join(snippet.lower().split())[:160])
        if key in seen:
            continue

        seen.add(key)
        unique_sources.append({
            "page": source.get("page"),
            "snippet": snippet[:180],
        })

    return unique_sources[:5]


def _normalize_response(response: dict, retrieved_chunks: list[dict]) -> dict:
    response.setdefault("answer", "")
    response.setdefault("key_insights", [])
    response.setdefault("risks", [])
    response.setdefault("recommendation", "")
    response.setdefault("sources", [])

    if not isinstance(response["key_insights"], list):
        response["key_insights"] = [str(response["key_insights"])]
    if not isinstance(response["risks"], list):
        response["risks"] = [str(response["risks"])]
    if not isinstance(response["sources"], list) or not response["sources"]:
        response["sources"] = [
            {
                "page": chunk.get("page"),
                "snippet": chunk.get("text", "")[:180],
            }
            for chunk in retrieved_chunks
        ]

    response["sources"] = _dedupe_source_items(response["sources"])
    return response


def _fallback_response(raw_output: str, retrieved_chunks: list[dict]) -> dict:
    raw_output = raw_output.strip()
    if raw_output.startswith("{"):
        raw_output = "The model returned an incomplete structured response. Please retry the question."

    return {
        "answer": raw_output,
        "key_insights": [],
        "risks": [],
        "recommendation": "",
        "sources": [
            {
                "page": chunk.get("page"),
                "snippet": chunk.get("text", "")[:180],
            }
            for chunk in retrieved_chunks
        ],
    }


def _parse_llm_json(raw_output: str, retrieved_chunks: list[dict]) -> dict:
    try:
        return _normalize_response(json.loads(raw_output), retrieved_chunks)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")

        if start != -1 and end != -1 and end > start:
            try:
                return _normalize_response(json.loads(raw_output[start:end + 1]), retrieved_chunks)
            except json.JSONDecodeError:
                pass

    return _fallback_response(raw_output, retrieved_chunks)


def _build_general_chat_prompt(
    message: str,
    history: list[dict] | None = None,
    context_chunks: list[dict] | None = None,
) -> str:
    history_lines = []
    for item in history or []:
        role = item.get("role", "user")
        content = item.get("content", "")
        if content:
            history_lines.append(f"{role}: {content}")

    context_lines = []
    for i, chunk in enumerate(context_chunks or [], start=1):
        context_lines.append(
            f"[Source {i} | Page {chunk.get('page')}]\n{chunk.get('text', '')}"
        )

    return f"""
{CHAT_SYSTEM_PROMPT}

Conversation history:
{chr(10).join(history_lines) if history_lines else "No prior conversation."}

Knowledge-base context:
{chr(10).join(context_lines) if context_lines else "No relevant uploaded context available."}

User message:
{message}

Respond in polished plain text. Do not return JSON.
"""


def _retrieve_chat_context(message: str, limit: int = 3) -> list[dict]:
    if len(vector_store.metadata) == 0:
        return []

    query_embedding = embed_texts_gemini([message], task_type="RETRIEVAL_QUERY")[0]
    _ensure_vector_store_dim(len(query_embedding))
    retrieved_chunks = vector_store.search(query_embedding, k=RETRIEVAL_CANDIDATES)
    retrieved_chunks = [chunk for _, chunk in _rerank_chunks(message, retrieved_chunks)]
    retrieved_chunks = _dedupe_chunks(retrieved_chunks)
    return _limit_chunks(retrieved_chunks, limit=limit, max_chars=600)


async def process_document(file: UploadFile):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = Path(file.filename or "uploaded.pdf").name
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    text_data = extract_text_from_pdf(file_path)
    chunks = chunk_text(text_data)

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts_gemini(texts, task_type="RETRIEVAL_DOCUMENT")

    if not embeddings:
        return {"chunks": 0}

    _reset_vector_store(len(embeddings[0]))
    vector_store.add(embeddings, chunks)
    vector_store.save()

    return {"chunks": len(chunks)}


async def query_rag(query: str):
    query_embedding = embed_texts_gemini([query], task_type="RETRIEVAL_QUERY")[0]
    _ensure_vector_store_dim(len(query_embedding))

    retrieved_chunks = vector_store.search(query_embedding, k=RETRIEVAL_CANDIDATES)
    retrieved_chunks = [chunk for _, chunk in _rerank_chunks(query, retrieved_chunks)]
    retrieved_chunks = _dedupe_chunks(retrieved_chunks)
    retrieved_chunks = _limit_chunks(retrieved_chunks)

    prompt = build_prompt(retrieved_chunks, query)

    print(f"query: {query}")
    print(f"retrieved_chunks: {len(retrieved_chunks)}")
    print(f"prompt_length: {len(prompt)}")

    raw_output = call_gemini_llm(
        prompt,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        max_output_tokens=2048,
    )

    print(f"response_length: {len(raw_output)}")

    return _parse_llm_json(raw_output, retrieved_chunks)


async def chat_with_assistant(
    message: str,
    history: list[dict] | None = None,
    use_knowledge_base: bool = True,
) -> dict:
    retrieved_chunks = []

    if use_knowledge_base:
        retrieved_chunks = _retrieve_chat_context(message)

    prompt = _build_general_chat_prompt(message, history, retrieved_chunks)

    print(f"chat_message: {message}")
    print(f"chat_context_chunks: {len(retrieved_chunks)}")
    print(f"chat_prompt_length: {len(prompt)}")

    answer = call_gemini_llm(
        prompt,
        response_mime_type="text/plain",
        max_output_tokens=1200,
    )

    print(f"chat_response_length: {len(answer)}")

    return {
        "answer": answer.strip(),
        "sources": [
            {
                "page": chunk.get("page"),
                "snippet": chunk.get("text", "")[:180],
            }
            for chunk in retrieved_chunks
        ],
        "used_knowledge_base": bool(retrieved_chunks),
        "model": _effective_gemini_model(),
    }


def stream_chat_with_assistant(
    message: str,
    history: list[dict] | None = None,
    use_knowledge_base: bool = True,
):
    retrieved_chunks = _retrieve_chat_context(message) if use_knowledge_base else []
    prompt = _build_general_chat_prompt(message, history, retrieved_chunks)

    print(f"stream_chat_message: {message}")
    print(f"stream_chat_context_chunks: {len(retrieved_chunks)}")
    print(f"stream_chat_prompt_length: {len(prompt)}")

    yield {
        "type": "metadata",
        "model": _effective_gemini_model(),
        "used_knowledge_base": bool(retrieved_chunks),
        "sources": [
            {
                "page": chunk.get("page"),
                "snippet": chunk.get("text", "")[:180],
            }
            for chunk in retrieved_chunks
        ],
    }

    response_length = 0
    for token in call_gemini_llm_stream(prompt):
        response_length += len(token)
        yield {
            "type": "token",
            "content": token,
        }

    print(f"stream_chat_response_length: {response_length}")
    yield {
        "type": "done",
    }
