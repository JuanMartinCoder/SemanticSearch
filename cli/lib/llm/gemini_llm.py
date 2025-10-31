
import json
import os
from time import sleep
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder


load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")


class GeminiLLM:
    def __init__(self, api_key: str = API_KEY, model: str = "gemini-2.0-flash") -> None:
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def enhance_request(self, query: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=query,
        )
        return response


def generate_enhanced_query(query: str, method: str) -> str:
    match method:
        case "spell":
            new_query = f"""Fix any spelling errors in this movie search query.
                    Only correct obvious typos. Don't change correctly spelled words.
                    Query: "{query}"
                    If no errors, return the original query.
                    Corrected:"""
            return new_query
        case "rewrite":
            new_query = f"""Rewrite this movie search query to be more specific and searchable.

                        Original: "{query}"

                        Consider:
                        - Common movie knowledge (famous actors, popular films)
                        - Genre conventions (horror = scary, animation = cartoon)
                        - Keep it concise (under 10 words)
                        - It should be a google style search query that's very specific

                        Rewritten query:"""
            return new_query
        case "expand":
            new_query = f"""Expand this movie search query with related terms.

                        Add synonyms and related concepts that might appear in movie descriptions.
                        Keep expansions relevant and focused.
                        This will be appended to the original query.

                        Example: "scary bear movie" -> "horror grizzly terrifying film" (3-5 more terms)


                        Query: "{query}"
                        """
            return new_query
        case _:
            raise ValueError(f"Invalid method: {method}")

       
def enhance_query(query: str, method: str = "spell") -> str:
    gemini = GeminiLLM()
    new_query = generate_enhanced_query(query, method)
    response = gemini.enhance_request(new_query, method)
    return response.text


def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    
    client = GeminiLLM()
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.

        Score:
        """

        response = client.enhance_request(prompt)
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "rerank_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_docs[:limit]

def llm_rerank_batch(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    if not documents:
        return []
    

    doc_map = {}
    doc_list = []
    for doc in documents:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)

    client = GeminiLLM()
    prompt = f"""Rank these movies by relevance to the search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

        [75, 12, 34, 2, 1]
        """

    response = client.enhance_request(prompt)
    ranking_text = (response.text or "").strip()
    parsed_ids = json.loads(ranking_text)

    reranked = []
    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            reranked.append({**doc_map[doc_id], "batch_rank": i + 1})


    return reranked[:limit]

def llm_rerank_cross_encoder(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    if not documents:
        return []
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
    

    # scores is a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    documents.sort(key=lambda x: x["rerank_score"], reverse=True)
    return documents[:limit]
   
def llm_judge_results(query: str, results: list[dict]) -> list[int]:
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(f"{i}. {result['title']}")

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

            Query: "{query}"

            Results:
            {chr(10).join(formatted_results)}

            Scale:
            - 3: Highly relevant
            - 2: Relevant
            - 1: Marginally relevant
            - 0: Not relevant

            Do NOT give any numbers out than 0, 1, 2, or 3.

            Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

            [2, 0, 3, 2, 0, 1]"""
    
    client = GeminiLLM()
    response = client.enhance_request(prompt)
    scores_text = (response.text or "").strip()
    scores = json.loads(scores_text)

    if len(scores) == len(results):
        return list(map(int, scores))

    raise ValueError(
        f"LLM response parsing error. Expected {len(results)} scores, got {len(scores)}. Response: {scores}"
    )


def rerank(query: str, documents: list[dict], method: str = "batch", limit: int = 5) -> list[dict]:
    match method:
        case "batch":
            return llm_rerank_batch(query, documents, limit)
        case "individual":
            return llm_rerank_individual(query, documents, limit)
        case "cross_encoder":
            return llm_rerank_cross_encoder(query, documents, limit)
        case _:
            raise documents[:limit]
