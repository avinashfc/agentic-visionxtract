"""
API router for face extraction endpoints.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
from modules.face_extraction.workflows.face_extraction_workflow import FaceExtractionWorkflow
from modules.face_extraction.models.face_extraction import (
    FaceExtractionRequest,
    FaceExtractionResponse,
    ExtractedFace
)

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
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")
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
    workflow: FaceExtractionWorkflow = Depends(get_workflow)
):
    """
    Extract faces from uploaded document.
    
    Args:
        file: Uploaded document file
        min_confidence: Minimum confidence threshold (0.0-1.0)
        extract_all_faces: Whether to extract all faces or just the first
        
    Returns:
        FaceExtractionResponse with extracted faces
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
            extract_all_faces=extract_all_faces
        )
        
        # Pydantic will automatically serialize image_data to base64 via field_serializer
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post("/extract-faces-batch")
async def extract_faces_batch(
    files: list[UploadFile] = File(..., description="List of document image files"),
    min_confidence: float = 0.7,
    extract_all_faces: bool = True,
    workflow: FaceExtractionWorkflow = Depends(get_workflow)
):
    """
    Extract faces from multiple uploaded documents.
    
    Args:
        files: List of uploaded document files
        min_confidence: Minimum confidence threshold (0.0-1.0)
        extract_all_faces: Whether to extract all faces or just the first
        
    Returns:
        List of FaceExtractionResponse objects
    """
    try:
        documents = []
        for file in files:
            file_content = await file.read()
            max_size = 10 * 1024 * 1024  # 10MB
            if len(file_content) > max_size:
                continue  # Skip oversized files
            documents.append((file_content, file.filename or "unknown"))
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents provided")
        
        # Execute batch workflow
        responses = await workflow.execute_batch(
            documents=documents,
            min_confidence=min_confidence,
            extract_all_faces=extract_all_faces
        )
        
        # Pydantic will automatically serialize image_data to base64 via field_serializer
        return {"results": responses, "total_processed": len(responses)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for face extraction service."""
    return {"status": "healthy", "service": "face_extraction"}


@router.post("/debug-detect")
async def debug_face_detection(
    file: UploadFile = File(..., description="Document image file"),
    min_confidence: float = 0.1,  # Lower threshold for debugging
    workflow: FaceExtractionWorkflow = Depends(get_workflow)
):
    """
    Debug endpoint to test face detection with detailed output.
    
    Args:
        file: Uploaded document file
        min_confidence: Minimum confidence threshold (default: 0.1 for debugging)
        
    Returns:
        Detailed debug information about face detection
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Get the face detector directly
        face_detector = workflow.agent.face_detector if hasattr(workflow.agent, 'face_detector') else workflow.agent.tools.face_detector
        
        # Try to detect faces with lower threshold
        face_detections = await face_detector.detect_faces(
            file_content,
            min_confidence=min_confidence,
            max_results=10
        )
        
        return {
            "file_name": file.filename,
            "file_size": len(file_content),
            "min_confidence_used": min_confidence,
            "faces_detected": len(face_detections),
            "face_details": [
                {
                    "x": face.x,
                    "y": face.y,
                    "width": face.width,
                    "height": face.height,
                    "confidence": face.confidence
                }
                for face in face_detections
            ]
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

