"""
Pydantic models for LLM Judge module.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EvaluationCriteria(BaseModel):
    """Criteria for evaluation."""
    name: str = Field(..., description="Name of the criteria (e.g., 'accuracy', 'relevance')")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Weight of this criteria (0.0-1.0)")
    description: Optional[str] = Field(None, description="Description of what this criteria evaluates")


class Score(BaseModel):
    """Score for a specific criteria."""
    criteria: str = Field(..., description="Name of the criteria")
    score: float = Field(..., ge=0.0, le=1.0, description="Score value (0.0-1.0)")
    reasoning: str = Field(..., description="Reasoning for this score")
    weight: float = Field(default=1.0, description="Weight of this criteria")


class JudgeRequest(BaseModel):
    """Request for LLM judge evaluation."""
    content: str = Field(..., description="Content to evaluate")
    reference: Optional[str] = Field(None, description="Reference content for comparison")
    criteria: Optional[List[EvaluationCriteria]] = Field(
        None,
        description="Custom evaluation criteria. If not provided, uses default criteria."
    )
    task_description: Optional[str] = Field(
        None,
        description="Description of the task being evaluated"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context for evaluation"
    )


class JudgeResponse(BaseModel):
    """Response from LLM judge evaluation."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall score (0.0-1.0)")
    scores: List[Score] = Field(..., description="Scores for each criteria")
    reasoning: str = Field(..., description="Overall reasoning for the evaluation")
    strengths: List[str] = Field(default_factory=list, description="List of strengths identified")
    weaknesses: List[str] = Field(default_factory=list, description="List of weaknesses identified")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")
    evaluation_id: str = Field(..., description="Unique identifier for this evaluation")


class ComparisonRequest(BaseModel):
    """Request for comparing multiple outputs."""
    outputs: List[str] = Field(..., min_items=2, description="List of outputs to compare")
    criteria: Optional[List[EvaluationCriteria]] = Field(
        None,
        description="Custom evaluation criteria. If not provided, uses default criteria."
    )
    task_description: Optional[str] = Field(
        None,
        description="Description of the task being evaluated"
    )
    rank: bool = Field(default=True, description="Whether to rank the outputs")


class ComparisonResult(BaseModel):
    """Result for a single output in comparison."""
    output_index: int = Field(..., description="Index of the output in the comparison")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall score")
    scores: List[Score] = Field(..., description="Scores for each criteria")
    reasoning: str = Field(..., description="Reasoning for this output")
    rank: Optional[int] = Field(None, description="Rank (1 = best, higher = worse)")


class ComparisonResponse(BaseModel):
    """Response from LLM judge comparison."""
    results: List[ComparisonResult] = Field(..., description="Results for each output")
    best_output_index: int = Field(..., description="Index of the best output")
    summary: str = Field(..., description="Summary of the comparison")
    comparison_id: str = Field(..., description="Unique identifier for this comparison")

