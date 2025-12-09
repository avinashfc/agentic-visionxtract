"""
Workflow for LLM judge evaluation and comparison.
"""
import logging
from typing import Optional, List, Dict, Any
from google.adk import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from modules.llm_judge.agents.judge_agent import JudgeAgent
from modules.llm_judge.models.judge import (
    JudgeRequest,
    JudgeResponse,
    ComparisonRequest,
    ComparisonResponse,
    EvaluationCriteria
)

logger = logging.getLogger(__name__)


class JudgeWorkflow:
    """
    Workflow for LLM judge evaluation and comparison.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize judge workflow.
        
        Args:
            api_key: Google API key
            model_name: Gemini model name (overrides config if provided)
        """
        self.agent = JudgeAgent(
            api_key=api_key,
            model_name=model_name
        )
        # Runner is owned by workflow for scalability/reuse
        self.runner = Runner(
            app_name=self.agent.app_name,
            agent=self.agent.agent,
            session_service=InMemorySessionService()
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
        
        # Call tools directly (tools handle LLM calls)
        return await self.agent.tools.evaluate(
            content=request.content,
            criteria=request.criteria,
            reference=request.reference,
            task_description=request.task_description,
            context=request.context
        )
    
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
        
        # Call tools directly (tools handle LLM calls)
        return await self.agent.tools.compare(
            outputs=request.outputs,
            criteria=request.criteria,
            task_description=request.task_description,
            rank=request.rank
        )
