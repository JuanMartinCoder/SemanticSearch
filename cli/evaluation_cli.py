import argparse
import json

from lib.search_utils import GOLDEN_DATASET_PATH, RRF_K, load_movies
from lib.search.hybrid_search import HybridSearch, rrf_search_command
from lib.search.semantic_search import SemanticSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    golden_dataset = {}
    # run evaluation logic here
    with open(GOLDEN_DATASET_PATH, "r") as f:
        golden_dataset = json.load(f)

    result = evaluate_command(limit, golden_dataset)

    print(f"k={args.limit}\n")
    for query, res in result["results"].items():
        print(f"- Query: {query}")
        print(f"  - Precision@{args.limit}: {res['precision']:.4f}")
        print(f"  - Recall@{args.limit}: {res['recall']:.4f}")
        print(f"  - F1 Score: {res['f1_score']:.4f}")
        print(f"  - Retrieved: {', '.join(res['retrieved'])}")
        print(f"  - Relevant: {', '.join(res['relevant'])}")
        print()
    

def precision_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k

def recall_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / len(relevant_docs)

def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_command(limit: int = 5, golden_dataset: dict = {}) -> dict:

    movies = load_movies()
    test_cases = golden_dataset["test_cases"]

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    
    total_precision = 0
    results_by_query = {}
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])
        search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_docs = []
        for result in search_results:
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)

        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        recall = recall_at_k(retrieved_docs, relevant_docs, limit)
        f1 = f1_score(precision, recall)

        results_by_query[query] = {
            "precision": precision,
            "retrieved": retrieved_docs[:limit],
            "recall": recall,
            "f1_score": f1,
            "relevant": list(relevant_docs),
        }

        total_precision += precision

    return {
        "test_cases_count": len(test_cases),
        "limit": limit,
        "results": results_by_query,
    }


if __name__ == "__main__":
    main()