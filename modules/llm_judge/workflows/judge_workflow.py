"""
Workflow for LLM judge evaluation and comparison.
"""
from typing import Optional, List, Dict, Any
from modules.llm_judge.agents.judge_agent import JudgeAgent
from modules.llm_judge.models.judge import (
    JudgeRequest,
    JudgeResponse,
    ComparisonRequest,
    ComparisonResponse
)


class JudgeWorkflow:
    """
    Workflow for LLM judge evaluation and comparison.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize judge workflow.
        
        Args:
            api_key: Google API key
            model_name: Gemini model name
        """
        self.agent = JudgeAgent(
            api_key=api_key,
            model_name=model_name
        )
    
    async def execute(
        self,
        content: str,
        reference: Optional[str] = None,
        criteria: Optional[List[Dict[str, Any]]] = None,
        task_description: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> JudgeResponse:
        """
        Execute judge evaluation workflow.
        
        Args:
            content: Content to evaluate
            reference: Reference content for comparison
            criteria: Custom evaluation criteria
            task_description: Description of the task
            context: Additional context
            
        Returns:
            JudgeResponse with evaluation results
        """
        from modules.llm_judge.models.judge import EvaluationCriteria
        
        # Convert criteria dicts to EvaluationCriteria objects if provided
        criteria_objs = None
        if criteria:
            criteria_objs = [
                EvaluationCriteria(**c) if isinstance(c, dict) else c
                for c in criteria
            ]
        
        request = JudgeRequest(
            content=content,
            reference=reference,
            criteria=criteria_objs,
            task_description=task_description,
            context=context
        )
        
        return await self.agent.evaluate(request)
    
    async def execute_comparison(
        self,
        outputs: List[str],
        criteria: Optional[List[Dict[str, Any]]] = None,
        task_description: Optional[str] = None,
        rank: bool = True
    ) -> ComparisonResponse:
        """
        Execute comparison workflow.
        
        Args:
            outputs: List of outputs to compare
            criteria: Custom evaluation criteria
            task_description: Description of the task
            rank: Whether to rank outputs
            
        Returns:
            ComparisonResponse with comparison results
        """
        from modules.llm_judge.models.judge import EvaluationCriteria
        
        # Convert criteria dicts to EvaluationCriteria objects if provided
        criteria_objs = None
        if criteria:
            criteria_objs = [
                EvaluationCriteria(**c) if isinstance(c, dict) else c
                for c in criteria
            ]
        
        request = ComparisonRequest(
            outputs=outputs,
            criteria=criteria_objs,
            task_description=task_description,
            rank=rank
        )
        
        return await self.agent.compare(request)

