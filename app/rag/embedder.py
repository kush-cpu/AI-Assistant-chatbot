from app.rag.pipeline import embed_texts_gemini


def embed_texts(texts):
    return embed_texts_gemini(texts, task_type="RETRIEVAL_DOCUMENT")
