import logging
from typing import Optional, List, Dict, Any
from modules.ocr.agents.ocr_agent import OCRAgent
from modules.ocr.models.ocr import KeyValueResponse
from core.module_client import ModuleClient, CommunicationMode

logger = logging.getLogger(__name__)


class OCRWorkflow:
    """ADK-based workflow orchestrator for OCR and key-value extraction pipeline."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize OCR workflow using ADK agentic flow.
        
        Args:
            api_key: Google API key
            model_name: Gemini model name
        """
        self.agent = OCRAgent(
            api_key=api_key,
            model_name=model_name
        )
    
    async def execute(
        self,
        file_content: bytes,
        document_name: str,
        language_hints: Optional[List[str]] = None,
        extraction_prompt: Optional[str] = None,
        evaluate_with_judge: bool = False,
        judge_criteria: Optional[List[Dict[str, Any]]] = None,
        judge_task_description: Optional[str] = None
    ) -> KeyValueResponse:
        """
        Execute the OCR and key-value extraction workflow.
        
        Args:
            file_content: Binary content of the document
            document_name: Name of the document
            language_hints: Optional list of language codes to help detection
            extraction_prompt: Optional custom prompt for key-value extraction
            evaluate_with_judge: Whether to evaluate key-value extraction result using LLM Judge module
            judge_criteria: Optional custom criteria for evaluation
            judge_task_description: Optional task description for judge
            
        Returns:
            KeyValueResponse (with evaluation added if evaluate_with_judge=True)
        """
        # Perform key-value extraction using agent
        response = await self.agent.extract_key_value_pairs(
            file_content=file_content,
            document_name=document_name,
            language_hints=language_hints,
            extraction_prompt=extraction_prompt
        )
        
        # Optionally evaluate with judge module via A2A
        if evaluate_with_judge and response.key_value_pairs:
            try:
                # Convert key-value pairs to a readable format for evaluation
                kv_text = "\n".join([
                    f"{kv.key}: {kv.value}" 
                    for kv in response.key_value_pairs
                ])
                
                async with ModuleClient("llm_judge", mode=CommunicationMode.AUTO) as judge_client:
                    evaluation = await judge_client.evaluate(
                        content=kv_text,
                        criteria=judge_criteria,
                        task_description=judge_task_description or f"Evaluate key-value extraction quality for document: {document_name}",
                        context={
                            "document_name": document_name,
                            "language_hints": language_hints,
                            "key_value_pairs_count": len(response.key_value_pairs),
                            "raw_text_length": len(response.raw_text)
                        }
                    )
                    
                    # Add evaluation to response metadata
                    if not hasattr(response, 'metadata'):
                        response.metadata = {}
                    response.metadata['evaluation'] = evaluation
                    response.metadata['evaluated'] = True
            except Exception as e:
                # Log error but don't fail extraction - evaluation is optional
                logger.warning(f"Failed to evaluate key-value extraction result with judge: {e}")
                if not hasattr(response, 'metadata'):
                    response.metadata = {}
                response.metadata['evaluation_error'] = str(e)
                response.metadata['evaluated'] = False
        
        return response

