#!/usr/bin/env python3

import argparse
from lib.keyword_search import (
    bm25_idf_command,
    bm25_tf_command,
    bm25search_command,
    buil_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)
from lib.search_utils import BM25_B, BM25_K1

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    
    subparsers.add_parser("build", help="Build inverted index")    
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a given document ID and term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get Inverse document frequency from a given term"
    )
    idf_parser.add_argument("term", type=str, help="Term to get frequency for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score for a given document ID and term"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get frequency for")

    bm25_idf_parser = subparsers.add_parser(
      'bm25idf', help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
   "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
                
        case "build":
            print("Building inverted index...")
            buil_command()
            print("Inverted index built successfully!.")

        case "tf":
            print("Getting term frequency for:", args.doc_id, args.term)
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency for {args.term} in document {args.doc_id}: {tf}")
        case "idf":
            print("Getting inverse document frequency for:", args.term)
            idf = idf_command(args.term)
            print(f"Inverse document frequency for {args.term}: {idf:.2f}")
        case "tfidf":
            print("Getting TF-IDF score for:", args.doc_id, args.term)
            tfidf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score for {args.term} in document {args.doc_id}: {tfidf:.2f}")
        case "bm25idf":
            print("Getting BM25 IDF score for:", args.term)
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            print("Getting BM25 TF score for:", args.doc_id, args.term)
            bm25_tf = bm25_tf_command(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
        case "bm25search":
            print("Searching for:", args.query)
            results = bm25search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
