import pdfplumber
import math
import requests
from google import genai
from google.genai import types
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN_LIMIT_LEGAL = 128_000
MODEL_NAME_LEGAL = "gemma-3-27b-it"


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
        return {"status": "error", "message": str(e), "explanation": None}
