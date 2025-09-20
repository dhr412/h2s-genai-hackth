from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
from infer import process_legal_document

app = FastAPI(
    title="Legal Document Analysis API",
    description="API for analyzing legal documents",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LegalDocumentRequest(BaseModel):
    pdf_url: HttpUrl

    class Config:
        json_schema_extra = {
            "example": {"pdf_url": "https://example.com/legal-document.pdf"}
        }


class LegalDocumentResponse(BaseModel):
    status: str
    explanation: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}


@app.post("/analyze-legal-document", response_model=LegalDocumentResponse)
async def analyze_legal_document(request: LegalDocumentRequest):
    try:
        pdf_url = str(request.pdf_url)

        result = process_legal_document(pdf_url)

        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        return LegalDocumentResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Legal Document Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "legal_analysis": "/analyze-legal-document",
        },
    }

