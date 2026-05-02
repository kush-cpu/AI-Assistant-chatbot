# AI Decision Assistant API

Production-oriented FastAPI backend for an AI decision assistant that answers user questions with uploaded document context, citations, risk reasoning, and streamed chat responses.

The frontend is built separately in Lovable. This repository contains the backend service deployed on Railway.

## What This Project Does

The system lets a user upload a PDF document, converts that document into searchable chunks, stores embeddings in a FAISS vector index, and uses Gemini to answer questions with relevant document context.

It supports two main interaction modes:

- **Document Q&A**: ask a specific question against the uploaded document and receive structured JSON with answer, key insights, risks, recommendation, and source snippets.
- **Streaming chat**: send normal chat messages from the Lovable frontend and receive streamed text responses over Server-Sent Events. When enabled, the assistant also retrieves relevant uploaded document chunks and includes source metadata.

## Architecture

```text
User
  -> Lovable Frontend
  -> Railway-hosted FastAPI Backend
  -> RAG Pipeline
  -> FAISS Vector Store
  -> Gemini Embeddings + Gemini LLM
  -> Streaming / JSON Response
  -> Lovable UI
```

### Backend Components

```text
app/main.py
  FastAPI app setup, CORS configuration, router registration.

app/routes.py
  Public API routes, request models, streaming response wrapper, legacy routes.

app/rag/processor.py
  PDF text extraction, text cleaning, sentence-aware chunking, chunk overlap.

app/rag/pipeline.py
  Gemini API calls, embeddings, retrieval, reranking, prompt construction, JSON parsing,
  general chat, streaming chat, and document processing.

app/rag/prompt.py
  Grounded RAG prompt used for structured document Q&A.

app/rag/embedder.py
  Compatibility wrapper around the Gemini embedding function.

app/rag/retriever.py
  Compatibility retrieval helper using the vector store.

app/storage/vector_store.py
  FAISS index creation, search, save, and load.
```

## Core Features

- PDF upload and server-side processing.
- Text extraction with PyMuPDF.
- Chunking with overlap to preserve context across chunk boundaries.
- Gemini embedding generation for uploaded chunks and user queries.
- FAISS vector search for fast local retrieval.
- Lightweight reranking based on query term overlap.
- Structured RAG answers with citations, key insights, risks, and recommendations.
- General assistant chat with optional knowledge-base retrieval.
- SSE streaming endpoint for frontend token-by-token style responses.
- CORS support for Lovable or any external frontend.
- Railway deployment support through `Procfile`.
- Legacy `/upload` and `/query` routes for older clients.

## Tech Stack

- **Backend**: FastAPI
- **Server**: Uvicorn
- **LLM**: Gemini generateContent API
- **Embeddings**: Gemini embedding API
- **Vector Store**: FAISS
- **PDF Processing**: PyMuPDF
- **Frontend**: Lovable
- **Deployment**: Railway

## Local Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env`

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
FRONTEND_ORIGINS=*
```

For production, replace `FRONTEND_ORIGINS=*` with the Lovable frontend domain.

Example:

```env
FRONTEND_ORIGINS=https://your-lovable-app.lovable.app
```

### 4. Run locally

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Base API URL:

```text
http://127.0.0.1:8000/api/v1
```

## Railway Deployment

Railway starts the application using the included `Procfile`:

```text
web: python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Set these Railway environment variables:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
FRONTEND_ORIGINS=https://your-lovable-app.lovable.app
```

After deployment, use the Railway service URL as the backend base URL in Lovable.

## API Overview

### Root

```http
GET /
```

Returns service status, docs URL, and endpoint links.

### Health Check

```http
GET /api/v1/health
```

Response:

```json
{
  "status": "ok"
}
```

### Capabilities

```http
GET /api/v1/capabilities
```

Returns endpoint metadata that the frontend can use to discover available backend capabilities.

### Upload Document

```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data
```

Form field:

```text
file: PDF file
```

Example response:

```json
{
  "message": "Document processed",
  "document": {
    "filename": "case-study.pdf",
    "chunks": 42
  }
}
```

What happens internally:

1. The PDF is saved under `data/uploads/`.
2. Text is extracted page by page.
3. Text is cleaned and split into overlapping chunks.
4. Chunks are embedded with Gemini.
5. Embeddings and metadata are saved to FAISS under `data/index/`.

### Ask Uploaded Document

```http
POST /api/v1/rag/query
Content-Type: application/json
```

Request:

```json
{
  "question": "What are the main risks discussed in the document?"
}
```

Example response shape:

```json
{
  "answer": "A grounded answer based on the retrieved document context.",
  "key_insights": [
    "Insight with page reference."
  ],
  "risks": [
    "Risk or limitation visible in the source context."
  ],
  "recommendation": "Recommended next step based only on the document.",
  "sources": [
    {
      "page": 3,
      "snippet": "Short supporting snippet..."
    }
  ]
}
```

This endpoint is best for document-specific decision questions where the frontend wants structured data for citations, insights, risks, and recommendations.

### General Chat

```http
POST /api/v1/chat
Content-Type: application/json
```

Request:

```json
{
  "message": "Summarize the uploaded document in simple terms.",
  "history": [
    {
      "role": "user",
      "content": "Previous user message"
    },
    {
      "role": "assistant",
      "content": "Previous assistant response"
    }
  ],
  "use_knowledge_base": true
}
```

Example response:

```json
{
  "answer": "Plain-text assistant answer.",
  "sources": [
    {
      "page": 1,
      "snippet": "Relevant uploaded context..."
    }
  ],
  "used_knowledge_base": true,
  "model": "gemini-2.5-flash"
}
```

This endpoint is useful when the frontend wants a normal non-streamed chat response.

### Streaming Chat

```http
POST /api/v1/chat/stream
Content-Type: application/json
Accept: text/event-stream
```

Request body is the same as `/api/v1/chat`:

```json
{
  "message": "Explain the decision tradeoffs from the document.",
  "history": [],
  "use_knowledge_base": true
}
```

The response is streamed as Server-Sent Events:

```text
data: {"type":"metadata","model":"gemini-2.5-flash","used_knowledge_base":true,"sources":[...]}

data: {"type":"token","content":"The"}

data: {"type":"token","content":" document"}

data: {"type":"done"}
```

Possible event types:

- `metadata`: sent first; includes model, source snippets, and whether document context was used.
- `token`: streamed text content.
- `done`: marks the end of the response.
- `error`: returned if the backend encounters an exception while streaming.

This is the main endpoint used by the Lovable frontend for live assistant responses.

## Lovable Frontend Integration

The Lovable frontend should call the Railway backend URL.

Recommended frontend flow:

1. Upload PDF with `POST /api/v1/documents/upload`.
2. Store chat messages in Lovable state.
3. Send messages to `POST /api/v1/chat/stream`.
4. Render `metadata.sources` as citations.
5. Append each `token.content` event to the assistant message.
6. Stop loading when `done` arrives.

For structured document-specific screens, use `POST /api/v1/rag/query` and render:

- `answer`
- `key_insights`
- `risks`
- `recommendation`
- `sources`

## RAG Pipeline Details

### Document Processing

`process_document()` in `app/rag/pipeline.py` handles uploaded files.

It:

- saves the uploaded PDF
- extracts text with `extract_text_from_pdf()`
- chunks extracted text with `chunk_text()`
- embeds each chunk with Gemini
- resets the active FAISS index
- saves the new vector index and metadata

### Retrieval

For a user query:

1. The query is embedded with Gemini using `RETRIEVAL_QUERY`.
2. FAISS returns the nearest chunks.
3. Chunks are reranked with a lightweight query-term overlap heuristic.
4. Duplicate chunks are removed.
5. The top chunks are trimmed to a maximum context size.

### Generation

For `/api/v1/rag/query`, the backend uses a strict JSON response schema. The prompt instructs the model to:

- answer only from retrieved context
- cite page numbers
- include risks and recommendations
- return `Insufficient data` when the document context does not support an answer
- return valid JSON only

For `/api/v1/chat` and `/api/v1/chat/stream`, the backend uses a plain-text assistant prompt with optional retrieved document context.

## Data Storage

Generated local data is stored under:

```text
data/uploads/
data/index/
```

These paths are ignored by Git.

The vector store persists:

- FAISS index: `data/index/faiss.index`
- chunk metadata: `data/index/metadata.pkl`

Current behavior is optimized for one active knowledge base at a time. Uploading a new document resets the vector store.

## Observability

The backend currently logs operational details to standard output, which Railway captures in deployment logs.

Logged information includes:

- query text
- number of retrieved chunks
- prompt length
- response length
- streaming response length
- number of context chunks used in chat

Recommended production improvements:

- replace `print()` with structured logging
- include request IDs
- record retrieval latency, embedding latency, LLM latency, and total latency
- log retrieval scores and selected source pages
- add error-level logs for failed Gemini calls

## Evaluation Approach

Use a small evaluation set of uploaded documents and expected questions.

Suggested checks:

- Does the answer stay grounded in retrieved context?
- Are citations present and relevant?
- Does the system say `Insufficient data` when the answer is not in the document?
- Are risks and recommendations actually supported by the source?
- Does streaming complete without malformed SSE events?

Example evaluation table:

| Question | Expected Behavior | Pass Criteria |
| --- | --- | --- |
| "What is the main decision?" | Summarizes only document-backed decision context. | Answer cites source page. |
| "What are the risks?" | Lists risks from retrieved chunks. | Risks match document text. |
| "Who is the CEO?" when absent | Refuses to guess. | Says insufficient data. |
| "Summarize the document" | Uses multiple relevant chunks. | Includes source snippets. |

Prompt comparison can be added by running the same evaluation questions against multiple prompt versions and comparing groundedness, completeness, and citation quality.

## Assignment Requirement Mapping

| Requirement | Status | Notes |
| --- | --- | --- |
| Document upload | Implemented | PDF upload supported. |
| Text upload | Not yet implemented | Current processor is PDF-focused. |
| Chunking and embeddings | Implemented | Sentence-aware chunks with overlap and Gemini embeddings. |
| Store metadata | Implemented | Page number and text stored with chunks. |
| Retrieve relevant chunks | Implemented | FAISS search plus lightweight reranking. |
| Generate answers with citations | Implemented | Structured RAG endpoint returns source snippets and pages. |
| Reasoning, risks, conclusions | Implemented | `risks`, `recommendation`, and `key_insights` returned by RAG endpoint. |
| Upload API | Implemented | `/api/v1/documents/upload`. |
| Query API | Implemented | `/api/v1/rag/query`. |
| History API | Partial | Chat accepts frontend-provided history, but backend does not persist history. |
| Chat UI | External | Built in Lovable. |
| Streaming responses | Implemented | `/api/v1/chat/stream` uses SSE. |
| Citation display | Frontend-dependent | Backend sends source metadata for Lovable to render. |
| Observability | Partial | Railway/stdout logs exist; structured latency logging is future work. |
| Evaluation | Planned | Manual evaluation approach documented; automated harness not yet implemented. |
| GitHub repo | Implemented | Repository includes backend source and deployment files. |
| README | Implemented | This file documents setup, architecture, endpoints, and tradeoffs. |
| Demo video | External | Record using Lovable frontend and Railway backend. |

## Design Tradeoffs

### Lovable frontend instead of in-repo React

The frontend is managed in Lovable to move quickly on UI, streaming interaction, and deployment. This repository stays focused on the backend API and RAG system.

### FAISS instead of managed vector database

FAISS keeps the project simple, fast, and cheap for a demo or assignment environment. A production multi-user version would likely use a managed vector store or database-backed document index.

### Single active document index

The current backend resets the vector store on each upload. This makes the behavior predictable for demos, but it does not yet support multiple users or multiple document collections.

### Client-provided chat history

The backend accepts history in each chat request rather than storing conversations. This works well with Lovable state, but a production system would persist conversation history server-side.

### Standard output logging

Railway captures stdout logs, so simple logging is enough for development. Production observability should use structured logs and latency metrics.

## Limitations

- Only PDF upload is currently supported.
- Uploaded documents replace the active vector index.
- No user accounts or document ownership model.
- No backend-persisted chat history.
- No automated evaluation harness yet.
- No cache layer yet.
- Retrieval uses a simple FAISS index and lightweight reranking, not a learned reranker.

## Security Notes

- Keep `.env` out of Git.
- Set `FRONTEND_ORIGINS` to the Lovable domain in production.
- Do not expose `GEMINI_API_KEY` to the frontend.
- Add authentication before using this with private or sensitive documents.
- Consider file size limits and file type validation for production.

## Project Structure

```text
.
|-- app/
|   |-- main.py
|   |-- routes.py
|   |-- rag/
|   |   |-- embedder.py
|   |   |-- pipeline.py
|   |   |-- processor.py
|   |   |-- prompt.py
|   |   `-- retriever.py
|   `-- storage/
|       `-- vector_store.py
|-- data/
|   |-- uploads/
|   `-- index/
|-- Procfile
|-- requirements.txt
`-- README.md
```

`data/uploads/` and `data/index/` are runtime directories and are ignored by Git.

## Quick Test Commands

Start the server:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Ask a non-streaming chat question:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"Hello, what can you do?\",\"history\":[],\"use_knowledge_base\":false}"
```

## Future Improvements

- Add `.txt` upload support.
- Add document IDs and support multiple document collections.
- Add backend-persisted chat history.
- Add structured logging and latency metrics.
- Add automated groundedness evaluation.
- Add authentication for private deployments.
- Add caching for repeated questions.
- Add stronger retrieval scoring and optional reranking.
