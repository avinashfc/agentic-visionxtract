"""
API router for OCR endpoints.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Optional
from modules.ocr.workflows.ocr_workflow import OCRWorkflow
from modules.ocr.models.ocr import KeyValueResponse

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


router = APIRouter()

# Initialize workflow (can be dependency injected in production)
_ocr_workflow: Optional[OCRWorkflow] = None


def get_ocr_workflow() -> OCRWorkflow:
    """Get or create OCR workflow instance."""
    global _ocr_workflow
    if _ocr_workflow is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL_NAME")  # Optional override, config.yaml is primary source
        _ocr_workflow = OCRWorkflow(
            api_key=api_key,
            model_name=model_name
        )
    return _ocr_workflow


@router.post("/extract-key-value-pairs", response_model=KeyValueResponse)
async def extract_key_value_pairs_from_document(
    file: UploadFile = File(..., description="Document image file"),
    language_hints: Optional[str] = None,  # Comma-separated language codes
    extraction_prompt: Optional[str] = None,  # Custom prompt for extraction
    evaluate_with_judge: bool = False,  # Whether to evaluate key-value extraction result with LLM Judge
    workflow: OCRWorkflow = Depends(get_ocr_workflow)
):
    """
    Extract key-value pairs from uploaded document using OCR and LLM.
    Optionally evaluates the extraction result using LLM Judge module (A2A communication).
    
    Args:
        file: Uploaded document file
        language_hints: Optional comma-separated language codes (e.g., "en,es,fr")
        extraction_prompt: Optional custom prompt for key-value extraction
        evaluate_with_judge: If True, evaluates key-value extraction quality using LLM Judge module
        
    Returns:
        KeyValueResponse with extracted key-value pairs (and evaluation if evaluate_with_judge=True)
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
            extraction_prompt=extraction_prompt,
            evaluate_with_judge=evaluate_with_judge,
            judge_task_description=f"Evaluate key-value extraction quality for document: {file.filename or 'unknown'}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for OCR service."""
    return {"status": "healthy", "service": "ocr"}

