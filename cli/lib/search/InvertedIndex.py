

import os
import pickle
import math
from collections import Counter, defaultdict

from ..search_utils import (
    BM25_B,
    BM25_K1,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
    CACHE_PATH
)

from ..text_processing import (
    tokenize
)

class InvertedIndex:

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_freq = defaultdict(Counter)
        self.doc_lengths = {}

        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_freq_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_freq[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total_length = 0.0
        for doc_id in self.doc_lengths:
            total_length += self.doc_lengths[doc_id]
        return total_length / len(self.doc_lengths)


    def get_documents(self, term: str) -> list[int]:
        docs_id = self.index.get(term, set())
        return sorted(list(docs_id))
    
    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_freq_path, "wb") as f:
            pickle.dump(self.term_freq, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_freq_path, "rb") as f:
            self.term_freq = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id: str, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single word")
        token = tokens[0]
        return self.term_freq[doc_id][token]
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single word")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_tfidf(self, doc_id: str, term: str) -> float:
        tf_score = self.get_tf(doc_id, term)
        idf_score = self.get_idf(term)
        return tf_score * idf_score
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single word")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log(((doc_count - term_doc_count + 0.5)/(term_doc_count + 0.5)) + 1)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def bm25(self, doc_id:int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
    
    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        tokens = tokenize(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)
        return results


        
            
