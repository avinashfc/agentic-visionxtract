"""
API router for OCR endpoints.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Optional, List
from modules.ocr.workflows.ocr_workflow import OCRWorkflow
from modules.ocr.workflows.key_value_workflow import KeyValueWorkflow
from modules.ocr.models.ocr import (
    OCRRequest,
    OCRResponse,
    KeyValueResponse
)

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


router = APIRouter()

# Initialize workflows (can be dependency injected in production)
_workflow: Optional[OCRWorkflow] = None
_key_value_workflow: Optional[KeyValueWorkflow] = None


def get_workflow() -> OCRWorkflow:
    """Get or create OCR workflow instance."""
    global _workflow
    if _workflow is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")
        _workflow = OCRWorkflow(
            api_key=api_key,
            model_name=model_name
        )
    return _workflow


def get_key_value_workflow() -> KeyValueWorkflow:
    """Get or create key-value extraction workflow instance."""
    global _key_value_workflow
    if _key_value_workflow is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")
        _key_value_workflow = KeyValueWorkflow(
            api_key=api_key,
            model_name=model_name
        )
    return _key_value_workflow


@router.post("/extract-text", response_model=OCRResponse)
async def extract_text_from_document(
    file: UploadFile = File(..., description="Document image file"),
    language_hints: Optional[str] = None,  # Comma-separated language codes
    workflow: OCRWorkflow = Depends(get_workflow)
):
    """
    Extract text from uploaded document using OCR.
    
    Args:
        file: Uploaded document file
        language_hints: Optional comma-separated language codes (e.g., "en,es,fr")
        
    Returns:
        OCRResponse with extracted text
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
            language_hints=language_list
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post("/extract-text-batch")
async def extract_text_batch(
    files: list[UploadFile] = File(..., description="List of document image files"),
    language_hints: Optional[str] = None,
    workflow: OCRWorkflow = Depends(get_workflow)
):
    """
    Extract text from multiple uploaded documents.
    
    Args:
        files: List of uploaded document files
        language_hints: Optional comma-separated language codes
        
    Returns:
        List of OCRResponse objects
    """
    try:
        documents = []
        for file in files:
            file_content = await file.read()
            max_size = 10 * 1024 * 1024  # 10MB
            if len(file_content) > max_size:
                continue
            documents.append((file_content, file.filename or "unknown"))
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents provided")
        
        # Parse language hints
        language_list = None
        if language_hints:
            language_list = [lang.strip() for lang in language_hints.split(",") if lang.strip()]
        
        responses = await workflow.execute_batch(
            documents=documents,
            language_hints=language_list
        )
        
        return {"results": responses, "total_processed": len(responses)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")


@router.post("/extract-key-value-pairs", response_model=KeyValueResponse)
async def extract_key_value_pairs_from_document(
    file: UploadFile = File(..., description="Document image file"),
    language_hints: Optional[str] = None,  # Comma-separated language codes
    extraction_prompt: Optional[str] = None,  # Custom prompt for extraction
    workflow: KeyValueWorkflow = Depends(get_key_value_workflow)
):
    """
    Extract key-value pairs from uploaded document using OCR and LLM.
    
    Args:
        file: Uploaded document file
        language_hints: Optional comma-separated language codes (e.g., "en,es,fr")
        extraction_prompt: Optional custom prompt for key-value extraction
        
    Returns:
        KeyValueResponse with extracted key-value pairs
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
            language_hints=language_list,
            extraction_prompt=extraction_prompt
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post("/extract-key-value-pairs-batch")
async def extract_key_value_pairs_batch(
    files: list[UploadFile] = File(..., description="List of document image files"),
    language_hints: Optional[str] = None,
    extraction_prompt: Optional[str] = None,
    workflow: KeyValueWorkflow = Depends(get_key_value_workflow)
):
    """
    Extract key-value pairs from multiple uploaded documents.
    
    Args:
        files: List of uploaded document files
        language_hints: Optional comma-separated language codes
        extraction_prompt: Optional custom prompt for key-value extraction
        
    Returns:
        List of KeyValueResponse objects
    """
    try:
        documents = []
        for file in files:
            file_content = await file.read()
            max_size = 10 * 1024 * 1024  # 10MB
            if len(file_content) > max_size:
                continue
            documents.append((file_content, file.filename or "unknown"))
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents provided")
        
        # Parse language hints
        language_list = None
        if language_hints:
            language_list = [lang.strip() for lang in language_hints.split(",") if lang.strip()]
        
        responses = await workflow.execute_batch(
            documents=documents,
            language_hints=language_list,
            extraction_prompt=extraction_prompt
        )
        
        return {"results": responses, "total_processed": len(responses)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for OCR service."""
    return {"status": "healthy", "service": "ocr"}

