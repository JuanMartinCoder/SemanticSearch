

import json
import os

import numpy as np

from ..search_utils import CACHE_PATH, CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH, DEFAULT_CHUNK_OVERLAP, DEFAULT_DOCUMENT_PREVIEW_SIZE, DEFAULT_SEARCH_LIMIT, DEFAULT_SEMANTIC_CHUNK_SIZE, format_search_result, load_movies
from ..search.semantic_search import SemanticSearch, cosine_similarity, semantic_chunk


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_path = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
        self.metadata_path = os.path.join(CACHE_PATH, "chunk_metadata.json")
    
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunk(
                text,
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
            )

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": idx, "chunk_idx": i, "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc
        self.documents = documents

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )

        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": i,
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": similarity,
                }
            )

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for movie_idx, score in sorted_movies[:limit]:
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DEFAULT_DOCUMENT_PREVIEW_SIZE],
                    score=score,
                )
            )

        return results

            
def embed_chunks() -> np.ndarray:
    movies = load_movies()
    model = ChunkedSemanticSearch()
    return model.load_or_create_chunk_embeddings(movies)
    

def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    searcher = ChunkedSemanticSearch()
    searcher.load_or_create_chunk_embeddings(movies)
    results = searcher.search_chunks(query, limit)
    return {"query": query, "results": results}