"""
API router for face extraction endpoints.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from typing import Optional, List, Dict, Any
from modules.face_extraction.workflows.face_extraction_workflow import FaceExtractionWorkflow
from modules.face_extraction.models.face_extraction import FaceExtractionResponse

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


router = APIRouter()

# Initialize workflow (can be dependency injected in production)
_workflow: Optional[FaceExtractionWorkflow] = None


def get_workflow() -> FaceExtractionWorkflow:
    """Get or create face extraction workflow instance."""
    global _workflow
    if _workflow is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL_NAME")  # Optional override, config.yaml is primary source
        _workflow = FaceExtractionWorkflow(
            api_key=api_key,
            model_name=model_name
        )
    return _workflow


@router.post("/extract-faces", response_model=FaceExtractionResponse)
async def extract_faces_from_document(
    file: UploadFile = File(..., description="Document image file"),
    min_confidence: float = 0.3,  # Lower default threshold for better detection
    extract_all_faces: bool = True,
    evaluate_with_judge: bool = Query(False, description="Whether to evaluate face extraction result using LLM Judge"),
    workflow: FaceExtractionWorkflow = Depends(get_workflow)
):
    """
    Extract faces from uploaded document.
    
    Args:
        file: Uploaded document file
        min_confidence: Minimum confidence threshold (0.0-1.0)
        extract_all_faces: Whether to extract all faces or just the first
        evaluate_with_judge: Whether to evaluate face extraction result using LLM Judge module
        
    Returns:
        FaceExtractionResponse with extracted faces (with evaluation added if evaluate_with_judge=True)
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {max_size} bytes"
            )
        
        # Execute workflow
        response = await workflow.execute(
            file_content=file_content,
            document_name=file.filename or "unknown",
            min_confidence=min_confidence,
            extract_all_faces=extract_all_faces,
            evaluate_with_judge=evaluate_with_judge
        )
        
        # Pydantic will automatically serialize image_data to base64 via field_serializer
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for face extraction service."""
    return {"status": "healthy", "service": "face_extraction"}
