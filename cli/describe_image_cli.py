import argparse
import mimetypes
import os
from google.genai import types

from google import genai

from dotenv import load_dotenv

model = "gemini-2.5-flash"
instructions = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""


def main():

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    parser = argparse.ArgumentParser(description="mage + text → rewritten query")
    parser.add_argument("--image", type=str, help="Search query for RAG")
    parser.add_argument("--query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, "rb") as f:
        img = f.read()

    client = genai.Client(api_key=api_key)
    parts = [
        instructions.strip(),
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    response = client.models.generate_content(model=model, contents=parts)
    if response.text is None:
        raise RuntimeError("No text in Gemini response")

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()