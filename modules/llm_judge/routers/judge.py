"""
API router for LLM Judge endpoints.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from modules.llm_judge.workflows.judge_workflow import JudgeWorkflow
from modules.llm_judge.models.judge import (
    JudgeRequest,
    JudgeResponse,
    ComparisonRequest,
    ComparisonResponse
)

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


router = APIRouter()

# Initialize workflow (can be dependency injected in production)
_workflow: Optional[JudgeWorkflow] = None


def get_workflow() -> JudgeWorkflow:
    """Get or create judge workflow instance."""
    global _workflow
    if _workflow is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL_NAME")  # Optional override, config.yaml is primary source
        _workflow = JudgeWorkflow(
            api_key=api_key,
            model_name=model_name
        )
    return _workflow


@router.post("/evaluate", response_model=JudgeResponse)
async def evaluate_content(
    request: JudgeRequest,
    workflow: JudgeWorkflow = Depends(get_workflow)
):
    """
    Evaluate content using LLM judge.
    
    Args:
        request: JudgeRequest with content and evaluation criteria
        
    Returns:
        JudgeResponse with scores, reasoning, and recommendations
    """
    try:
        response = await workflow.agent.evaluate(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating content: {str(e)}")


@router.post("/compare", response_model=ComparisonResponse)
async def compare_outputs(
    request: ComparisonRequest,
    workflow: JudgeWorkflow = Depends(get_workflow)
):
    """
    Compare multiple outputs using LLM judge.
    
    Args:
        request: ComparisonRequest with outputs and criteria
        
    Returns:
        ComparisonResponse with comparison results and ranking
    """
    try:
        response = await workflow.agent.compare(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing outputs: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for LLM Judge service."""
    return {"status": "healthy", "service": "llm_judge"}

