# AI Decision Assistant API

FastAPI backend for a Gemini-powered AI assistant with document upload, RAG Q&A, general chat, and SSE streaming.

## Local Run

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Base API URL:

```text
http://127.0.0.1:8000/api/v1
```

## Environment Variables

Create a `.env` file locally or set these in Railway:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
FRONTEND_ORIGINS=*
```

For production, set `FRONTEND_ORIGINS` to your frontend domain.

## API Endpoints

```text
GET  /api/v1/health
GET  /api/v1/capabilities
POST /api/v1/documents/upload
POST /api/v1/rag/query
POST /api/v1/chat
POST /api/v1/chat/stream
```

## Railway

Railway can start the app using the included `Procfile`:

```text
web: python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
