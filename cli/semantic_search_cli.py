#!/usr/bin/env python3

import argparse


from lib.search_utils import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, DEFAULT_SEARCH_LIMIT
from lib.semantic_search import chunk_text, embed_query_text, embed_text, verify_embeddings, verify_model,semantic_search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify model")


    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Embed text"
    )
    embed_text_parser.add_argument("term", type=str, help="Term")

    embed_text_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify Embeddings"
    )

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Embed query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query")
    
    search_parser = subparsers.add_parser(
        "search", help="Search for movies"
    )
    search_parser.add_argument("query", type=str, help="Query")
    search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit results")

    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk text"
    )
    chunk_parser.add_argument("query", type=str, help="Query")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=DEFAULT_CHUNK_SIZE, help="Limit chunk size")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=DEFAULT_OVERLAP, help="overlap")

    args = parser.parse_args()



    match args.command:
        case "verify":
            print("Verifying model...")
            verify_model()
            print("Model verified successfully!")
        case "embed_text":
            print("Embedding text...")
            embed_text(args.term)
            print("Text embedded successfully!")
        case "verify_embeddings":
            print("Verifying embeddings...")
            verify_embeddings()
            print("Embeddings verified successfully!")
        case "embedquery":
            print("Embedding query...")
            embed_query_text(args.query)
            print("Query embedded successfully!")
        case "search":
            print("Searching for:", args.query)
            results = semantic_search(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (score: {res['score']:.4f})\n{res['description'][:100]}...\n")
        case "chunk":
            print(f"Chunking {len(args.query)} characters")
            chunk_text(args.query, args.chunk_size, args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()