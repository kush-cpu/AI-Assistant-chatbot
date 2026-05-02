import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "data/index/faiss.index"
META_PATH = "data/index/metadata.pkl"

class VectorStore:
    def __init__(self, dim=1536):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, metadatas):
        self.index.add(np.array(embeddings).astype("float32")) # type: ignore
        self.metadata.extend(metadatas)

    def search(self, query_embedding, k=3):
        if len(self.metadata) == 0:
            return []

        k = min(k, len(self.metadata))
        D, I = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        ) # type: ignore
        return [self.metadata[i] for i in I[0] if i != -1]

    def save(self):
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.metadata = pickle.load(f)
