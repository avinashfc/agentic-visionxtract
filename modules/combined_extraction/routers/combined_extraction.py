"""
API router for combined extraction endpoints.
This module orchestrates communication with face extraction and OCR agents.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Optional
from modules.combined_extraction.workflows.combined_extraction_workflow import CombinedExtractionWorkflow
from modules.combined_extraction.models.combined_extraction import CombinedExtractionResponse

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


router = APIRouter()

# Initialize workflow (can be dependency injected in production)
_workflow: Optional[CombinedExtractionWorkflow] = None


def get_workflow() -> CombinedExtractionWorkflow:
    """Get or create combined extraction workflow instance."""
    global _workflow
    if _workflow is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")
        _workflow = CombinedExtractionWorkflow(
            api_key=api_key,
            model_name=model_name
        )
    return _workflow


@router.post("/extract-all", response_model=CombinedExtractionResponse)
async def extract_faces_ocr_and_key_value_pairs(
    file: UploadFile = File(..., description="Document image file"),
    min_confidence: float = 0.7,
    extract_all_faces: bool = True,
    language_hints: Optional[str] = None,  # Comma-separated language codes
    extraction_prompt: Optional[str] = None,  # Custom prompt for key-value extraction
    workflow: CombinedExtractionWorkflow = Depends(get_workflow)
):
    """
    Combined extraction: Extract faces, perform OCR, and extract key-value pairs from uploaded document.
    
    This endpoint orchestrates communication with multiple agents:
    1. Face extraction agent: Detects and extracts faces from the document
    2. OCR agent: Extracts all text from the document
    3. Key-value extraction agent: Uses LLM to extract structured key-value pairs from OCR text
    
    Args:
        file: Uploaded document file
        min_confidence: Minimum confidence threshold for face detection (default: 0.7)
        extract_all_faces: Whether to extract all faces or just the first (default: True)
        language_hints: Optional comma-separated language codes (e.g., "en,es,fr")
        extraction_prompt: Optional custom prompt for key-value extraction
        
    Returns:
        CombinedExtractionResponse with faces, OCR text, and key-value pairs
    """
    try:
        file_content = await file.read()
        
        max_size = 10 * 1024 * 1024  # 10MB
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {max_size} bytes"
            )
        
        # Parse language hints
        language_list = None
        if language_hints:
            language_list = [lang.strip() for lang in language_hints.split(",") if lang.strip()]
        
        response = await workflow.execute(
            file_content=file_content,
            document_name=file.filename or "unknown",
            min_confidence=min_confidence,
            extract_all_faces=extract_all_faces,
            language_hints=language_list,
            extraction_prompt=extraction_prompt
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for combined extraction service."""
    return {"status": "healthy", "service": "combined-extraction"}

