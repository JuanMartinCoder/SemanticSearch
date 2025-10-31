import argparse

from lib.llm.augment_rag import llm_citations, llm_question, llm_summarize, rag
from lib.search.hybrid_search import rrf_search_command
from lib.search_utils import RRF_K


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summary_parser = subparsers.add_parser(
        "summarize", help="Summarize a query"
    )
    summary_parser.add_argument("query", type=str, help="Search query for RAG")
    summary_parser.add_argument("--limit", type=int,default=5, nargs='?', help="Search query for RAG")

    citation_parser = subparsers.add_parser(
        "citations", help="Citations for a query"
    )
    citation_parser.add_argument("query", type=str, help="Search query for RAG")
    citation_parser.add_argument("--limit", type=int,default=5, nargs='?', help="Search query for RAG")

    question_parser = subparsers.add_parser(
        "question", help="Question for a query"
    )
    question_parser.add_argument("query", type=str, help="Search query for RAG")
    question_parser.add_argument("--limit", type=int,default=5, nargs='?', help="Search query for RAG")
    
    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            result = rrf_search_command(
                args.query, RRF_K, limit=5
            )
            print("Search Results:")
            for res in result["results"]:
                print(f"- {res['title']}")
            
            print("RAG Response:")
            rag_response = rag(query, result["results"])
            print(rag_response.text)
        case "summarize":
            result = rrf_search_command(
                args.query, RRF_K, limit=args.limit
            )
            print("Search Results:")
            for res in result["results"]:
                print(f"- {res['title']}")

            print("LLM Summary:")
            llm_summary = llm_summarize(args.query, result["results"])
            print(llm_summary.text)

        case "citations":
            result = rrf_search_command(
                args.query, RRF_K, limit=args.limit
            )
            print("Search Results:")
            for res in result["results"]:
                print(f"- {res['title']}")

            print("LLM Answer:")
            llm_citation = llm_citations(args.query, result["results"])
            print(llm_citation.text)
        case "question":
            result = rrf_search_command(
                args.query, RRF_K, limit=args.limit
            )
            print("Search Results:")
            for res in result["results"]:
                print(f"- {res['title']}")

            print("Answer:")
            llm_question_res = llm_question(args.query, result["results"])
            print(llm_question_res.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()