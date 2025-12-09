"""
ADK-based agent for LLM judging and evaluation.
"""
from typing import Optional, List, Dict, Any
from google.adk import Agent, Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from modules.llm_judge.tools.judge_tools import JudgeTools
from modules.llm_judge.models.judge import (
    JudgeRequest,
    JudgeResponse,
    ComparisonRequest,
    ComparisonResponse
)
import logging

logger = logging.getLogger(__name__)


class JudgeAgent:
    """ADK-based agent for LLM judging and evaluation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize ADK judge agent.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        self.tools = JudgeTools(api_key=api_key, model_name=model_name)
        
        self.agent = self._create_agent()
        
        self.runner = Runner(
            app_name="judge_app",
            agent=self.agent,
            session_service=InMemorySessionService()
        )
    
    def _create_agent(self) -> Agent:
        """Create ADK Agent instance with judge tools."""
        agent_tools = self.tools.get_tools()
        
        agent_config = {
            "name": "judge_agent",
            "description": "Agent for evaluating and judging content using LLM with structured criteria",
            "model": self.model_name,
            "tools": agent_tools,
        }
        
        agent = Agent(**agent_config)
        return agent
    
    async def evaluate(
        self,
        request: JudgeRequest
    ) -> JudgeResponse:
        """
        Evaluate content using judge agent.
        
        Args:
            request: JudgeRequest with content and criteria
            
        Returns:
            JudgeResponse with scores and reasoning
        """
        return await self.tools.evaluate(
            content=request.content,
            criteria=request.criteria,
            reference=request.reference,
            task_description=request.task_description,
            context=request.context
        )
    
    async def compare(
        self,
        request: ComparisonRequest
    ) -> ComparisonResponse:
        """
        Compare multiple outputs using judge agent.
        
        Args:
            request: ComparisonRequest with outputs and criteria
            
        Returns:
            ComparisonResponse with comparison results
        """
        return await self.tools.compare(
            outputs=request.outputs,
            criteria=request.criteria,
            task_description=request.task_description,
            rank=request.rank
        )

