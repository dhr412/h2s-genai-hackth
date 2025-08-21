from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
from infer import process_legal_document, verify_misinformation

app = FastAPI(
    title="Legal Document & Misinformation Analysis API",
    description="API for analyzing legal documents and verifying misinformation claims",
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


class MisinformationRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {"example": {"query": "are aliens real"}}


class LegalDocumentResponse(BaseModel):
    status: str
    explanation: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SourceInfo(BaseModel):
    title: str
    url: str


class MisinformationResponse(BaseModel):
    status: str
    classification: Optional[str] = None
    explanation: Optional[str] = None
    detailed_explanation: Optional[str] = None
    sources: Optional[List[SourceInfo]] = None
    sources_count: Optional[int] = None
    message: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
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


@app.post("/verify-misinformation", response_model=MisinformationResponse)
async def verify_misinformation_endpoint(request: MisinformationRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        result = verify_misinformation(request.query.strip())

        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        if result.get("sources"):
            result["sources"] = [SourceInfo(**source) for source in result["sources"]]

        return MisinformationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Legal Document & Misinformation Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "legal_analysis": "/analyze-legal-document",
            "misinformation_check": "/verify-misinformation",
            "docs": "/docs",
        },
    }
