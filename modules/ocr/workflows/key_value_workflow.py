"""
Workflow for key-value extraction using Google ADK.
"""
import asyncio
from typing import Optional, List
from modules.ocr.agents.key_value_agent import KeyValueAgent
from modules.ocr.models.ocr import (
    KeyValueResponse
)


class KeyValueWorkflow:
    """ADK-based workflow orchestrator for key-value extraction pipeline."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize key-value extraction workflow using ADK agentic flow.
        
        Args:
            api_key: Google API key
            model_name: Gemini model name
        """
        self.agent = KeyValueAgent(
            api_key=api_key,
            model_name=model_name
        )
    
    async def execute(
        self,
        file_content: bytes,
        document_name: str,
        language_hints: Optional[List[str]] = None,
        extraction_prompt: Optional[str] = None
    ) -> KeyValueResponse:
        """
        Execute the key-value extraction workflow.
        
        Args:
            file_content: Binary content of the document
            document_name: Name of the document
            language_hints: Optional list of language codes to help detection
            extraction_prompt: Optional custom prompt for key-value extraction
            
        Returns:
            KeyValueResponse
        """
        response = await self.agent.extract_key_value_pairs(
            file_content=file_content,
            document_name=document_name,
            language_hints=language_hints,
            extraction_prompt=extraction_prompt
        )
        
        return response
    
    async def execute_batch(
        self,
        documents: list[tuple[bytes, str]],
        language_hints: Optional[List[str]] = None,
        extraction_prompt: Optional[str] = None
    ) -> list[KeyValueResponse]:
        """
        Execute key-value extraction workflow for multiple documents.
        
        Args:
            documents: List of tuples (file_content, document_name)
            language_hints: Optional language hints
            extraction_prompt: Optional custom prompt for key-value extraction
            
        Returns:
            List of KeyValueResponse objects
        """
        tasks = [
            self.execute(
                file_content=content,
                document_name=name,
                language_hints=language_hints,
                extraction_prompt=extraction_prompt
            )
            for content, name in documents
        ]
        
        responses = await asyncio.gather(*tasks)
        return responses


