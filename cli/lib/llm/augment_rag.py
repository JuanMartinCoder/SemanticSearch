

from .gemini_llm import API_KEY, GeminiLLM


class RAG(GeminiLLM):
    def __init__(self, api_key: str = API_KEY, model: str = "gemini-2.0-flash") -> None:
        super().__init__(api_key, model)
        self.model = model
        self.client = GeminiLLM(api_key, model)
        self.query = None
        self.results = None
    

    def enhance_prompt_rag(self, query: str, docs) -> str:
        prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {docs}

        Provide a comprehensive answer that addresses the query:"""
        return prompt 
    def enhance_prompt_summary(self, query: str, results) -> str:
        prompt = f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:
        {results}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
        """
        return prompt
    def enhance_prompt_citations(self, query: str, results) -> str:
        prompt = f"""Answer the question or provide information based on the provided documents.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

            Query: {query}

            Documents:
            {results}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources using [1], [2], etc. format when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
        return prompt
    def enhance_prompt_question(self, query: str, results) -> str:
        prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Question: {query}

        Documents:
        {results}

        Instructions:
        - Answer questions directly and concisely
        - Be casual and conversational
        - Don't be cringe or hype-y
        - Talk like a normal person would in a chat conversation

        Answer:"""
        return prompt

    def make_request(self, query: str, docs, method: str = "rag") -> str:
        self.query = query
        response = []
        if method == "rag":
            response = self.client.enhance_request(self.enhance_prompt_rag(query, docs))
        if method == "summary":
            response = self.client.enhance_request(self.enhance_prompt_summary(query, docs))
        if method == "citations":
            response = self.client.enhance_request(self.enhance_prompt_citations(query, docs))
        if method == "question":
            response = self.client.enhance_request(self.enhance_prompt_question(query, docs))
        self.results = response
        return response
    



def rag(query: str, docs: list[dict]) -> str:
    rag = RAG()

    results = []
    for doc in docs:
        results.append(
            f"{doc["id"]}: {doc["title"]}"
        )
    
    doc_list_str = "\n".join(results)

    return rag.make_request(query, doc_list_str, "rag")

def llm_summarize(query: str, docs: list[dict]) -> str:
    rag = RAG()
    results = []
    for doc in docs:
        results.append(
            f"{doc["id"]}: {doc["title"]}"
        )
    
    doc_list_str = "\n".join(results)

    return rag.make_request(query, doc_list_str, "summary")

def llm_citations(query: str, docs: list[dict]) -> str:
    rag = RAG()
    results = []
    for doc in docs:
        results.append(
            f"{doc["id"]}: {doc['title']}"
        )
    
    doc_list_str = "\n".join(results)
    return rag.make_request(query, doc_list_str, "citations")

def llm_question(query: str, docs: list[dict]) -> str:
    rag = RAG()
    results = []
    for doc in docs:
        results.append(
            f"{doc["id"]}: {doc['title']}"
        )
    doc_list_str = "\n".join(results)
    return rag.make_request(query, doc_list_str, "question")