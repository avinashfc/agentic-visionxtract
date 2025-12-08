"""
Workflow for face extraction using Google ADK.
"""
import asyncio
from typing import Optional
from modules.face_extraction.agents.face_extraction_agent import FaceExtractionAgent
from modules.face_extraction.models.face_extraction import (
    FaceExtractionRequest,
    FaceExtractionResponse
)


class FaceExtractionWorkflow:
    """ADK-based workflow orchestrator for face extraction pipeline - agentic flow only."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize face extraction workflow using ADK agentic flow.
        
        Args:
            api_key: Google API key
            model_name: Gemini model name
        """
        # Always use ADK agentic flow
        self.agent = FaceExtractionAgent(
            api_key=api_key,
            model_name=model_name
        )
    
    async def execute(
        self,
        file_content: bytes,
        document_name: str,
        min_confidence: float = 0.7,
        extract_all_faces: bool = True
    ) -> FaceExtractionResponse:
        """
        Execute the face extraction workflow.
        
        Args:
            file_content: Binary content of the document
            document_name: Name of the document
            min_confidence: Minimum confidence threshold
            extract_all_faces: Whether to extract all faces
            
        Returns:
            FaceExtractionResponse
        """
        # Execute the workflow through the agent
        response = await self.agent.process_document(
            file_content=file_content,
            document_name=document_name,
            min_confidence=min_confidence,
            extract_all_faces=extract_all_faces
        )
        
        return response
    
    async def execute_batch(
        self,
        documents: list[tuple[bytes, str]],
        min_confidence: float = 0.7,
        extract_all_faces: bool = True
    ) -> list[FaceExtractionResponse]:
        """
        Execute face extraction workflow for multiple documents.
        
        Args:
            documents: List of tuples (file_content, document_name)
            min_confidence: Minimum confidence threshold
            extract_all_faces: Whether to extract all faces
            
        Returns:
            List of FaceExtractionResponse objects
        """
        tasks = [
            self.execute(
                file_content=content,
                document_name=name,
                min_confidence=min_confidence,
                extract_all_faces=extract_all_faces
            )
            for content, name in documents
        ]
        
        responses = await asyncio.gather(*tasks)
        return responses

