import os
from typing import Optional

from lib.llm.gemini_llm import enhance_query, rerank

from ..search_utils import DEFAULT_SEARCH_LIMIT, RRF_K, SEARCH_MULTIPLIER, format_search_result, load_movies

from .keyword_search import InvertedIndex
from ..chunks.chunk_semantic import ChunkedSemanticSearch

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit=5) -> list[dict]:
        bm25_results = self._bm25_search(query, 500*limit)
        semantic_results = self.semantic_search.search_chunks(query, 500*limit)

        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query: str, k: int , limit=10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        fused = reciprocal_rank_fusion(bm25_results, semantic_results, k)
        return fused[:limit]
        
    

def reciprocal_rank_fusion(
    bm25_results: list[dict], semantic_results: list[dict], k: int = RRF_K
) -> list[dict]:
    rrf_scores = {}

    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["bm25_rank"] is None:
            rrf_scores[doc_id]["bm25_rank"] = rank
            score_contribution = rrf_score(rank, k)
            rrf_scores[doc_id]["rrf_score"] += score_contribution

    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["semantic_rank"] is None:
            rrf_scores[doc_id]["semantic_rank"] = rank
            score_contribution = rrf_score(rank, k)
            rrf_scores[doc_id]["rrf_score"] += score_contribution

    rrf_results = []
    for doc_id, data in rrf_scores.items():
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=data["rrf_score"],
            rrf_score=data["rrf_score"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        rrf_results.append(result)

    sorted_results = sorted(rrf_results, key=lambda x: x["score"], reverse=True)

    return sorted_results

def rrf_search_command(query: str, 
                       k: int = 60, 
                       enhance: Optional[str] = None,
                       rerank_method: Optional[str] = None,
                       limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query
    enhanced_query = None
        
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query
    
    search_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit
    results = searcher.rrf_search(query, k, search_limit)
    
    

    reranked = False
    if rerank_method:
        results = rerank(query, results, method=rerank_method, limit=limit)
        reranked = True

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "reranked": reranked,
        "results": results,
    }

def combine_search_results(bm25_results, semantic_results, alpha=0.5):
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    
    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_scores = []
    for doc_id, result in combined_scores.items():
        score_value = hybrid_score(result["bm25_score"], result["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=result["title"],
            document=result["document"],
            score=score_value,
            bm25_score=result["bm25_score"],
            semantic_score=result["semantic_score"],
        )
        hybrid_scores.append(result)

    return sorted(hybrid_scores, key=lambda x: x["score"], reverse=True)



def weighted_search_command(query: str, alpha: float = 0.5, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    
    searcher = HybridSearch(movies)

    original_query = query
    search_limit = limit

    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }


def normalize_args(args: list[float]) -> list[float]:
    if not args:
        return []

    max_arg = max(args)
    min_arg = min(args)

    if max_arg == min_arg:
        return [1.0] * len(args)

    normalized_args = []
    for arg in args:
        normalized_args.append((arg - min_arg) / (max_arg - min_arg))

    return normalized_args

def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_args(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

