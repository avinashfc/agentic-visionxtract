"""
Tools for LLM Judge agent.
"""
from typing import Optional, List, Dict, Any
from google.adk.tools import FunctionTool
from google.genai import Client, types
from modules.llm_judge.models.judge import (
    JudgeRequest,
    JudgeResponse,
    Score,
    EvaluationCriteria,
    ComparisonRequest,
    ComparisonResponse,
    ComparisonResult
)
import json
import uuid
import logging

logger = logging.getLogger(__name__)


class JudgeTools:
    """Tools for LLM judge functionality."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize judge tools.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name to use (must be provided)
        """
        if not model_name:
            raise ValueError("model_name must be provided. Configure it in config.yaml or pass it explicitly.")
        self.api_key = api_key
        self.model_name = model_name
        self.genai_client = None
        
        if api_key:
            try:
                self.genai_client = Client(api_key=api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")
    
    async def evaluate_content_tool(
        self,
        content: str,
        criteria: Optional[List[Dict[str, Any]]] = None,
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate content against criteria and return scores with reasoning.
        This is the tool function for ADK agent.
        
        Args:
            content: Content to evaluate
            criteria: Optional evaluation criteria (list of dicts with name, weight, description)
            reference: Optional reference content for comparison
            
        Returns:
            Dictionary with evaluation results
        """
        # Convert criteria dicts to EvaluationCriteria objects if provided
        criteria_objs = None
        if criteria:
            criteria_objs = [
                EvaluationCriteria(**c) if isinstance(c, dict) else c
                for c in criteria
            ]
        
        result = await self.evaluate(
            content=content,
            criteria=criteria_objs,
            reference=reference
        )
        
        # Convert to dict for tool response
        if hasattr(result, 'model_dump'):
            return result.model_dump(mode='json')
        elif hasattr(result, 'dict'):
            return result.dict()
        return result
    
    async def compare_outputs_tool(
        self,
        outputs: List[str],
        criteria: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple outputs and rank them.
        This is the tool function for ADK agent.
        
        Args:
            outputs: List of outputs to compare
            criteria: Optional evaluation criteria (list of dicts with name, weight, description)
            
        Returns:
            Dictionary with comparison results
        """
        # Convert criteria dicts to EvaluationCriteria objects if provided
        criteria_objs = None
        if criteria:
            criteria_objs = [
                EvaluationCriteria(**c) if isinstance(c, dict) else c
                for c in criteria
            ]
        
        result = await self.compare(
            outputs=outputs,
            criteria=criteria_objs
        )
        
        # Convert to dict for tool response
        if hasattr(result, 'model_dump'):
            return result.model_dump(mode='json')
        elif hasattr(result, 'dict'):
            return result.dict()
        return result
    
    def get_tools(self) -> List[FunctionTool]:
        """
        Get list of ADK FunctionTool instances.
        
        Returns:
            List of FunctionTool instances
        """
        tools = [
            FunctionTool(self.evaluate_content_tool),
            FunctionTool(self.compare_outputs_tool),
        ]
        
        return tools
    
    async def evaluate(
        self,
        content: str,
        criteria: Optional[List[EvaluationCriteria]] = None,
        reference: Optional[str] = None,
        task_description: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> JudgeResponse:
        """
        Evaluate content using LLM judge.
        
        Args:
            content: Content to evaluate
            criteria: Custom evaluation criteria
            reference: Reference content for comparison
            task_description: Description of the task
            context: Additional context
            
        Returns:
            JudgeResponse with scores and reasoning
        """
        if not self.genai_client:
            raise ValueError("Gemini client not initialized. Provide GOOGLE_API_KEY.")
        
        # Default criteria if not provided
        if not criteria:
            criteria = [
                EvaluationCriteria(name="accuracy", weight=0.3, description="Factual accuracy and correctness"),
                EvaluationCriteria(name="relevance", weight=0.25, description="Relevance to the task"),
                EvaluationCriteria(name="completeness", weight=0.2, description="Completeness of the response"),
                EvaluationCriteria(name="clarity", weight=0.15, description="Clarity and coherence"),
                EvaluationCriteria(name="quality", weight=0.1, description="Overall quality")
            ]
        
        # Build prompt
        criteria_text = "\n".join([
            f"- {c.name} (weight: {c.weight}): {c.description}"
            for c in criteria
        ])
        
        prompt = f"""You are an expert judge evaluating content. Evaluate the following content based on the criteria provided.

Task Description: {task_description or "General content evaluation"}

Evaluation Criteria:
{criteria_text}

Content to Evaluate:
{content}
"""
        
        if reference:
            prompt += f"\nReference Content (for comparison):\n{reference}\n"
        
        if context:
            prompt += f"\nAdditional Context:\n{json.dumps(context, indent=2)}\n"
        
        prompt += """
Please provide:
1. A score (0.0-1.0) for each criteria
2. Reasoning for each score
3. Overall score (weighted average)
4. Strengths identified
5. Weaknesses identified
6. Recommendations for improvement

Respond in JSON format with this structure:
{
    "scores": [
        {
            "criteria": "accuracy",
            "score": 0.85,
            "reasoning": "...",
            "weight": 0.3
        },
        ...
    ],
    "overall_score": 0.82,
    "reasoning": "Overall evaluation reasoning...",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "recommendations": ["recommendation1", "recommendation2"]
}
"""
        
        try:
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Lower temperature for more consistent judging
                    response_mime_type="application/json"
                )
            )
            
            result_text = response.text
            result_json = json.loads(result_text)
            
            # Parse scores
            scores = []
            for score_data in result_json.get("scores", []):
                # Find matching criteria
                criteria_obj = next(
                    (c for c in criteria if c.name == score_data.get("criteria")),
                    None
                )
                weight = criteria_obj.weight if criteria_obj else score_data.get("weight", 1.0)
                
                scores.append(Score(
                    criteria=score_data.get("criteria", "unknown"),
                    score=float(score_data.get("score", 0.0)),
                    reasoning=score_data.get("reasoning", ""),
                    weight=weight
                ))
            
            return JudgeResponse(
                overall_score=float(result_json.get("overall_score", 0.0)),
                scores=scores,
                reasoning=result_json.get("reasoning", ""),
                strengths=result_json.get("strengths", []),
                weaknesses=result_json.get("weaknesses", []),
                recommendations=result_json.get("recommendations", []),
                evaluation_id=str(uuid.uuid4())
            )
            
        except Exception as e:
            logger.error(f"Error in judge evaluation: {e}")
            raise
    
    async def compare(
        self,
        outputs: List[str],
        criteria: Optional[List[EvaluationCriteria]] = None,
        task_description: Optional[str] = None,
        rank: bool = True
    ) -> ComparisonResponse:
        """
        Compare multiple outputs using LLM judge.
        
        Args:
            outputs: List of outputs to compare
            criteria: Custom evaluation criteria
            task_description: Description of the task
            rank: Whether to rank the outputs
            
        Returns:
            ComparisonResponse with results for each output
        """
        if not self.genai_client:
            raise ValueError("Gemini client not initialized. Provide GOOGLE_API_KEY.")
        
        if len(outputs) < 2:
            raise ValueError("At least 2 outputs required for comparison")
        
        # Default criteria if not provided
        if not criteria:
            criteria = [
                EvaluationCriteria(name="accuracy", weight=0.3, description="Factual accuracy"),
                EvaluationCriteria(name="relevance", weight=0.25, description="Relevance to task"),
                EvaluationCriteria(name="completeness", weight=0.2, description="Completeness"),
                EvaluationCriteria(name="clarity", weight=0.15, description="Clarity"),
                EvaluationCriteria(name="quality", weight=0.1, description="Overall quality")
            ]
        
        # Build prompt
        criteria_text = "\n".join([
            f"- {c.name} (weight: {c.weight}): {c.description}"
            for c in criteria
        ])
        
        outputs_text = "\n\n".join([
            f"Output {i+1}:\n{output}"
            for i, output in enumerate(outputs)
        ])
        
        prompt = f"""You are an expert judge comparing multiple outputs. Evaluate and compare the following outputs based on the criteria provided.

Task Description: {task_description or "General content comparison"}

Evaluation Criteria:
{criteria_text}

Outputs to Compare:
{outputs_text}

Please evaluate each output and provide:
1. A score (0.0-1.0) for each criteria for each output
2. Reasoning for each output
3. Overall score for each output
4. Ranking (if requested)
5. Summary comparison

Respond in JSON format with this structure:
{{
    "results": [
        {{
            "output_index": 0,
            "overall_score": 0.85,
            "scores": [
                {{
                    "criteria": "accuracy",
                    "score": 0.9,
                    "reasoning": "...",
                    "weight": 0.3
                }},
                ...
            ],
            "reasoning": "Overall reasoning for this output...",
            "rank": 1
        }},
        ...
    ],
    "best_output_index": 0,
    "summary": "Summary of comparison..."
}}
"""
        
        try:
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            
            result_text = response.text
            result_json = json.loads(result_text)
            
            # Parse results
            results = []
            for result_data in result_json.get("results", []):
                scores = []
                for score_data in result_data.get("scores", []):
                    criteria_obj = next(
                        (c for c in criteria if c.name == score_data.get("criteria")),
                        None
                    )
                    weight = criteria_obj.weight if criteria_obj else score_data.get("weight", 1.0)
                    
                    scores.append(Score(
                        criteria=score_data.get("criteria", "unknown"),
                        score=float(score_data.get("score", 0.0)),
                        reasoning=score_data.get("reasoning", ""),
                        weight=weight
                    ))
                
                results.append(ComparisonResult(
                    output_index=int(result_data.get("output_index", 0)),
                    overall_score=float(result_data.get("overall_score", 0.0)),
                    scores=scores,
                    reasoning=result_data.get("reasoning", ""),
                    rank=int(result_data.get("rank", 0)) if rank else None
                ))
            
            return ComparisonResponse(
                results=results,
                best_output_index=int(result_json.get("best_output_index", 0)),
                summary=result_json.get("summary", ""),
                comparison_id=str(uuid.uuid4())
            )
            
        except Exception as e:
            logger.error(f"Error in judge comparison: {e}")
            raise

