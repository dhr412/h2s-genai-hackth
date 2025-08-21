import pdfplumber
import math
import requests
import time
from google import genai
from google.genai import types
from ddgs import DDGS
from bs4 import BeautifulSoup
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN_LIMIT_LEGAL = 128_000
TOKEN_LIMIT_MISINFO = 110_000
MODEL_NAME_LEGAL = "gemma-3-27b-it"
MODEL_NAME_MISINFO = "gemma-3n-e4b-it"


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)


def download_pdf(url: str, local_path: str = "temp_document.pdf") -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(response.content)
        return local_path
    except Exception as e:
        raise Exception(f"Failed to download PDF: {str(e)}")


def read_pdf_to_text(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                parts.append(txt)
    return "\n\n".join(parts)


def chunk_text(text: str, chunk_size: int = 30000) -> List[str]:
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = min(i + chunk_size, L)
        if end < L:
            next_space = text.find(" ", end)
            if next_space != -1 and next_space - i <= chunk_size * 1.2:
                end = next_space
        chunks.append(text[i:end].strip())
        i = end
    return [c for c in chunks if c]


def approx_tokens_from_chars(text: str, chars_per_token: float = 4.0) -> int:
    return max(1, math.ceil(len(text) / chars_per_token))


def process_legal_document(pdf_url: str) -> Dict[str, any]:
    try:
        client = get_gemini_client()

        local_pdf_path = download_pdf(pdf_url)

        pdf_text = read_pdf_to_text(local_pdf_path)

        if not pdf_text.strip():
            return {
                "status": "error",
                "message": "No text extracted from the PDF.",
                "explanation": None,
            }

        estimated_tokens = approx_tokens_from_chars(pdf_text)

        if estimated_tokens > TOKEN_LIMIT_LEGAL:
            chunks = chunk_text(pdf_text, chunk_size=30000)
        else:
            chunks = [pdf_text]

        all_responses = []
        for idx, chunk in enumerate(chunks, start=1):
            user_content = types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "You are a helpful assistant that explains legal documents in plain, "
                            "bite-sized bulleted chunks. Demystify technical/legal language while "
                            "preserving the legal meaning and any obligations. Flag ambiguous parts "
                            "or items that need human/legal review.\n\n"
                            f"Excerpt {idx} of {len(chunks)}:\n\n{chunk}\n\n"
                            "Task: Explain the excerpt in simple, bite-sized bullet points. "
                            "For each bullet, indicate if it references clause numbers, parties, deadlines, "
                            "or obligations. If a section requires special attention or a lawyer, say so. "
                            "Finish with a short plain-English summary of the excerpt."
                        )
                    )
                ],
            )

            response = client.models.generate_content(
                model=MODEL_NAME_LEGAL, contents=[user_content]
            )

            all_responses.append(response.text)

        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)

        combined_output = "\n\n".join(all_responses)

        return {
            "status": "success",
            "explanation": combined_output,
            "metadata": {
                "estimated_tokens": estimated_tokens,
                "chunks_processed": len(chunks),
                "within_token_limit": estimated_tokens <= TOKEN_LIMIT_LEGAL,
            },
        }

    except Exception as e:
        if "local_pdf_path" in locals() and os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)

        return {"status": "error", "message": str(e), "explanation": None}


def search_web_info(query: str, num_results: int = 8) -> List[Dict[str, str]]:
    search_results = []

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=num_results,
                    region="wt-wt",
                    safesearch="moderate",
                )
            )

            for result in results:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                    response = requests.get(result["href"], headers=headers, timeout=16)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.content, "html.parser")

                    for tag in soup(["script", "style", "nav", "header", "footer"]):
                        tag.decompose()

                    main_content = soup.find(["article", "main"]) or soup.find(
                        "div", class_=["content", "post"]
                    )
                    text = main_content.get_text() if main_content else soup.get_text()

                    clean_text = " ".join(text.split())[:3000]

                    search_results.append(
                        {
                            "title": result["title"],
                            "url": result["href"],
                            "content": f"{result['body']}\n\nFull content: {clean_text}",
                        }
                    )

                except Exception:
                    search_results.append(
                        {
                            "title": result["title"],
                            "url": result["href"],
                            "content": result["body"],
                        }
                    )

                time.sleep(0.5)

    except Exception as e:
        print(f"DDGS search failed: {str(e)}")

    return search_results


def extract_label(response_text: str) -> str:
    text = response_text.upper()

    if "TRUE" in text and "FALSE" not in text:
        return "TRUE"
    elif "FALSE" in text and "TRUE" not in text:
        return "FALSE"
    elif "PARTIALLY TRUE" in text or "PARTIAL" in text:
        return "PARTIALLY TRUE"
    elif "UNCERTAIN" in text or "UNCLEAR" in text:
        return "UNCERTAIN"
    else:
        return "UNCERTAIN"


def verify_misinformation(user_query: str) -> Dict[str, any]:
    try:
        client = get_gemini_client()

        search_results = search_web_info(user_query, num_results=5)

        if not search_results:
            return {
                "status": "error",
                "message": "No search results found. Cannot verify the information.",
                "classification": None,
                "explanation": None,
                "sources": [],
            }

        user_content = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        "You are a fact-checking assistant.\n\n"
                        f"Claim: {user_query}\n\n"
                        f"Sources:\n{search_results}\n\n"
                        "Task: Classify the claim as TRUE, FALSE, or PARTIALLY TRUE. "
                        "Explain the reasoning in clear, plain English, citing sources when possible."
                    )
                )
            ],
        )

        response = client.models.generate_content(
            model=MODEL_NAME_MISINFO, contents=[user_content]
        )

        classification = extract_label(response.text)

        detailed_explanation = None
        if classification == "FALSE":
            detail_content = types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Provide a detailed explanation of why the following claim might be misleading "
                            "or inaccurate. Reference the provided sources and highlight contradictions, "
                            "missing context, or manipulations.\n\n"
                            f"Claim: {user_query}\n\n"
                            f"Sources:\n{search_results}"
                        )
                    )
                ],
            )

            detail_response = client.models.generate_content(
                model=MODEL_NAME_MISINFO, contents=[detail_content]
            )

            detailed_explanation = detail_response.text

        return {
            "status": "success",
            "classification": classification,
            "explanation": response.text,
            "detailed_explanation": detailed_explanation,
            "sources": [{"title": r["title"], "url": r["url"]} for r in search_results],
            "sources_count": len(search_results),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "classification": None,
            "explanation": None,
            "sources": [],
        }
