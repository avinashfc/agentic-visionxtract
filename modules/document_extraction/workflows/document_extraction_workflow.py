"""
Workflow for document extraction using Google ADK.
This module orchestrates communication with face extraction and OCR agents.
"""
import time
from typing import Optional, List, Dict, Any
from modules.face_extraction.workflows.face_extraction_workflow import FaceExtractionWorkflow
from modules.ocr.workflows.ocr_workflow import OCRWorkflow
from modules.document_extraction.models.document_extraction import DocumentExtractionResponse


class DocumentExtractionWorkflow:
    """
    ADK-based workflow orchestrator for document extraction pipeline.
    
    This workflow communicates with:
    - FaceExtractionWorkflow (face extraction agent)
    - OCRWorkflow (OCR and key-value extraction agent)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize document extraction workflow using ADK agentic flow.
        
        Args:
            api_key: Google API key
            model_name: Gemini model name
        """
        # Initialize workflows that communicate with their respective agents
        self.face_workflow = FaceExtractionWorkflow(
            api_key=api_key,
            model_name=model_name
        )
        
        self.ocr_workflow = OCRWorkflow(
            api_key=api_key,
            model_name=model_name
        )
    
    async def execute(
        self,
        file_content: bytes,
        document_name: str,
        min_confidence: float = 0.7,
        extract_all_faces: bool = True,
        language_hints: Optional[List[str]] = None,
        extraction_prompt: Optional[str] = None,
        evaluate_with_judge: bool = False,
        judge_criteria: Optional[List[Dict[str, Any]]] = None,
        judge_task_description: Optional[str] = None
    ) -> DocumentExtractionResponse:
        """
        Execute the document extraction workflow: face extraction and key-value extraction.
        
        This method orchestrates communication with multiple agents:
        1. Face extraction agent (via FaceExtractionWorkflow)
        2. OCR and key-value extraction agent (via OCRWorkflow)
        
        Args:
            file_content: Binary content of the document
            document_name: Name of the document
            min_confidence: Minimum confidence threshold for face detection
            extract_all_faces: Whether to extract all faces
            language_hints: Optional list of language codes to help OCR detection
            extraction_prompt: Optional custom prompt for key-value extraction
            evaluate_with_judge: Whether to evaluate extraction results using LLM Judge module
            judge_criteria: Optional custom criteria for evaluation
            judge_task_description: Optional task description for judge
            
        Returns:
            DocumentExtractionResponse with all extraction results (with evaluation added if evaluate_with_judge=True)
        """
        start_time = time.time()
        
        # Step 1: Communicate with Face Extraction Agent
        face_start_time = time.time()
        face_response = await self.face_workflow.execute(
            file_content=file_content,
            document_name=document_name,
            min_confidence=min_confidence,
            extract_all_faces=extract_all_faces,
            evaluate_with_judge=evaluate_with_judge,
            judge_criteria=judge_criteria,
            judge_task_description=judge_task_description
        )
        face_extraction_time = time.time() - face_start_time
        
        # Step 2: Communicate with Key-Value Extraction Agent
        key_value_extraction_time = None
        key_value_pairs = []
        kv_response = None

        kv_start_time = time.time()
        kv_response = await self.ocr_workflow.execute(
            file_content=file_content,
            document_name=document_name,
            language_hints=language_hints,
            extraction_prompt=extraction_prompt,
            evaluate_with_judge=evaluate_with_judge,
            judge_criteria=judge_criteria,
            judge_task_description=judge_task_description
        )
        key_value_extraction_time = time.time() - kv_start_time
        key_value_pairs = kv_response.key_value_pairs
        
        total_time = time.time() - start_time
        
        # Determine overall status
        status_parts = []
        if face_response.faces_detected > 0:
            status_parts.append(f"{face_response.faces_detected} face(s) detected")
        if key_value_pairs:
            status_parts.append(f"{len(key_value_pairs)} key-value pair(s) extracted")
        
        status = "success: " + ", ".join(status_parts) if status_parts else "completed"
        
        # Convert ExtractedFace objects to dicts for JSON serialization
        # Using model_dump with mode='json' to ensure proper serialization of bytes to base64
        faces_extracted_dicts = [face.model_dump(mode='json') for face in face_response.faces_extracted]
        
        # Get document_id from any available response (they should all be the same)
        document_id = face_response.document_id
        if kv_response and hasattr(kv_response, 'document_id'):
            document_id = document_id or kv_response.document_id
        
        # Collect metadata from both responses if judge evaluation was performed
        metadata = {}
        if evaluate_with_judge:
            if hasattr(face_response, 'metadata') and face_response.metadata:
                metadata['face_extraction_evaluation'] = face_response.metadata.get('evaluation')
            if hasattr(kv_response, 'metadata') and kv_response.metadata:
                metadata['ocr_evaluation'] = kv_response.metadata.get('evaluation')
        
        return DocumentExtractionResponse(
            document_id=document_id,
            faces_detected=face_response.faces_detected,
            faces_extracted=faces_extracted_dicts,
            full_text=kv_response.raw_text,
            key_value_pairs=key_value_pairs,
            processing_time=total_time,
            face_extraction_time=face_extraction_time,
            key_value_extraction_time=key_value_extraction_time,
            status=status,
            metadata=metadata if metadata else None
        )
