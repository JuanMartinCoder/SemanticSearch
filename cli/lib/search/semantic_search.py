
import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np

from ..search_utils import CACHE_PATH, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, DEFAULT_SEARCH_LIMIT, DEFAULT_SEMANTIC_CHUNK_SIZE, format_search_result, load_movies

class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.cache_path = os.path.join(CACHE_PATH, "movie_embeddings.npy")

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        list_of_movies = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie = f"{doc['title']}: {doc['description']}"
            list_of_movies.append(movie)

        self.embeddings = self.model.encode(list_of_movies, show_progress_bar=True)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.save(self.cache_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.cache_path):
            self.embeddings = np.load(self.cache_path)
            if len(documents) == len(self.embeddings):
                return self.embeddings
        
        return self.build_embeddings(documents)

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        return self.model.encode([text])[0]
    
    def search(self, query, limit: int = DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )
        embedding_query = self.generate_embedding(query)
        similarity_scores = []
        for i, doc_emb in enumerate(self.embeddings):
            similarity = cosine_similarity(embedding_query, doc_emb)
            similarity_scores.append((similarity, self.documents[i]))
            
        similarity_scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarity_scores[:limit]:
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            })
        return results


def verify_model():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

def embed_text(text):
    model = SemanticSearch()
    emb = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {emb[:3]}")
    print(f"Dimensions: {emb.shape[0]}")  

def verify_embeddings():
    model = SemanticSearch()
    movies = load_movies()
    embeddings = model.load_or_create_embeddings(movies)
    print("Embeddings loaded successfully!")
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    model = SemanticSearch()
    embedding = model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    model = SemanticSearch()
    model.load_or_create_embeddings(load_movies())

    results = model.search(query, limit)
    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()
    return results

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
    words = text.split()
    textfull = ""
    j = 0
    if overlap == 0:
        for i in range(0, len(words), chunk_size):
            j+=1
            chunked_text = " ".join(words[i:i+chunk_size])
            textfull+=f"{j}. {chunked_text}\n"
    else:
        if overlap > 0:
            for i in range(0, len(words), chunk_size):
                j+=1
                if j > 1:
                    last_words = words[i-overlap:i]
                    chunked_text = " ".join(last_words) + " " + " ".join(words[i:i+chunk_size])
                else:
                    chunked_text = " ".join(words[i:i+chunk_size])
                textfull+=f"{j}. {chunked_text}\n"

    print(textfull)

def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    input = text.strip()
    if input == "":
        return []
    
    sentences = re.split(r'(?<=[.!?])\s+', input)

    if len(sentences) == 1 and not endsWithPrefix(input):
        sentences = [text]
    
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences - overlap:
        chunk_sentences = sentences[i : i + max_chunk_size]
        cleared_sentences = []
        for chunk_sentence in chunk_sentences:
            cleared_sentences.append(chunk_sentence.strip())
        if not cleared_sentences:
            continue
        chunk = " ".join(cleared_sentences)
        chunks.append(chunk)
        i += max_chunk_size - overlap

    return chunks


def chunk_semantic_text(text: str, chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP):
    chunks = semantic_chunk(text, chunk_size, overlap)
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")
    


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def endsWithPrefix(str: str) -> bool:
    return str.endswith(".") or str.endswith("!") or str.endswith("?")