import argparse
import logging
from lib.search.hybrid_search import normalize_args, rrf_search_command, weighted_search_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.llm.gemini_llm import llm_judge_results

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subaparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subaparser.add_parser("normalize", help="Normalize list of args")    
    normalize_parser.add_argument("list", type=float, nargs="+", default=[], help="list of inputs")

    normalize_parser = subaparser.add_parser("weighted-search", help="weighted search")
    normalize_parser.add_argument("query", type=str, help="query")
    normalize_parser.add_argument("--alpha", type=float, nargs='?', default=0.5 ,help="query")
    normalize_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT ,help="limit")

    rrf_parser = subaparser.add_parser(
        "rrf-search", help="Perform Reciprocal Rank Fusion search"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF k parameter controlling weight distribution (default=60)",
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "expand", "rewrite"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual","batch","cross_encoder"],
        help="Reranking method",
    )

    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluation mode",
    )
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            results = normalize_args(args.list)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res:.4f}")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)

            print(f"Weighted Hybrid Search Results for '{results['query']}' (alpha={results['alpha']}):")
            print(f"  Alpha {results['alpha']}: {int(results['alpha'] * 100)}% Keyword, {int((1 - results['alpha']) * 100)}% Semantic")
            for i, res in enumerate(results["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()

        case "rrf-search":
            result = rrf_search_command(
                args.query, args.k, args.enhance, args.rerank_method, args.limit
            )
            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )

            if result["reranked"]:
                print(
                    f"Reranking top {len(result['results'])} results using {result['rerank_method']} method...\n"
                )

            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):"
            )

            

            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                if result["rerank_method"] and "rerank_score" in res:
                    print(f"   Rerank Score: {res.get('rerank_score', 0):.3f}/10")
                if "individual_score" in res:
                    print(f"   Rerank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Rerank Rank: {res.get('batch_rank', 0)}")
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()

            if args.evaluate:
                print("LLM Evaluation (0-3 relevance scale):")

                llm_scores = llm_judge_results(args.query, result["results"])

                for i, (res, score) in enumerate(zip(result["results"], llm_scores), 1):
                    print(f"{i}. {res['title']}: {score}/3")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()