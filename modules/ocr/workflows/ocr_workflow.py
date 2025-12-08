"""
Workflow for OCR using Google ADK.
"""
import asyncio
from typing import Optional, List
from modules.ocr.agents.ocr_agent import OCRAgent
from modules.ocr.models.ocr import (
    OCRRequest,
    OCRResponse
)


class OCRWorkflow:
    """ADK-based workflow orchestrator for OCR pipeline - agentic flow only."""
    
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
        language_hints: Optional[List[str]] = None
    ) -> OCRResponse:
        """
        Execute the OCR workflow.
        
        Args:
            file_content: Binary content of the document
            document_name: Name of the document
            language_hints: Optional list of language codes to help detection
            
        Returns:
            OCRResponse
        """
        response = await self.agent.process_document(
            file_content=file_content,
            document_name=document_name,
            language_hints=language_hints
        )
        
        return response
    
    async def execute_batch(
        self,
        documents: list[tuple[bytes, str]],
        language_hints: Optional[List[str]] = None
    ) -> list[OCRResponse]:
        """
        Execute OCR workflow for multiple documents.
        
        Args:
            documents: List of tuples (file_content, document_name)
            language_hints: Optional language hints
            
        Returns:
            List of OCRResponse objects
        """
        tasks = [
            self.execute(
                file_content=content,
                document_name=name,
                language_hints=language_hints
            )
            for content, name in documents
        ]
        
        responses = await asyncio.gather(*tasks)
        return responses

