from app.storage.vector_store import VectorStore
from app.rag.embedder import embed_texts

vector_store = VectorStore()
vector_store.load()

def retrieve(query, k=3, store=None):
    active_store = store or vector_store
    query_embedding = embed_texts([query])[0]
    return active_store.search(query_embedding, k=k)
