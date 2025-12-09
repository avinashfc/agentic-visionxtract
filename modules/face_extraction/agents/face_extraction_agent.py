"""
ADK-based agent for face extraction from documents.
"""
import time
import base64
import uuid
from typing import Optional, Dict, Any
from google.adk import Agent, Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from modules.face_extraction.tools.face_extraction_tools import FaceExtractionTools
from modules.face_extraction.models.face_extraction import (
    FaceExtractionResponse,
    ExtractedFace
)
import logging

logger = logging.getLogger(__name__)


class FaceExtractionAgent:
    """ADK-based agent for extracting faces from uploaded documents."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize ADK face extraction agent.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        self.tools = FaceExtractionTools(api_key=api_key)
        
        self.agent = self._create_agent()
        
        self.runner = Runner(
            app_name="face_extraction_app",
            agent=self.agent,
            session_service=InMemorySessionService()
        )
    
    def _create_agent(self) -> Agent:
        """Create ADK Agent instance with tools."""
        agent_tools = self.tools.get_tools()
        
        agent_config = {
            "name": "face_extraction_agent",
            "description": "Agent for extracting faces from uploaded documents using Vision API",
            "model": self.model_name,
            "tools": agent_tools,
        }
        
        agent = Agent(**agent_config)
        
        return agent
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set context for tool execution."""
        self.tools._context = context
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context."""
        return getattr(self.tools, '_context', {})
    
    def clear_context(self) -> None:
        """Clear context."""
        if hasattr(self.tools, '_context'):
            delattr(self.tools, '_context')
    
    async def execute_task(
        self,
        task_prompt: str,
        user_id: str = "face_extraction_user"
    ) -> Dict[str, Any]:
        """
        Execute a task using the agent.
        
        Args:
            task_prompt: Task description for the agent
            user_id: User identifier for the session
            
        Returns:
            Dictionary with execution results and context
        """
        new_message = types.Content(parts=[types.Part(text=task_prompt)])
        session_id = str(uuid.uuid4())
        
        await self.runner.session_service.create_session(
            app_name=self.runner.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        events = []
        function_calls_executed = 0
        
        try:
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message
            ):
                events.append(event)
                
                if hasattr(event, 'get_function_calls'):
                    function_calls = event.get_function_calls()
                    if function_calls:
                        function_calls_executed += len(function_calls)
                        logger.debug(f"Agent executed {len(function_calls)} function call(s): {[fc.name for fc in function_calls]}")
            
            logger.debug(f"Agent execution completed - {len(events)} events, {function_calls_executed} function calls executed")
            
            return {
                "success": True,
                "events": events,
                "function_calls_executed": function_calls_executed,
                "context": self.get_context()
            }
        except Exception as e:
            logger.error(f"Agent execution error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "context": self.get_context()
            }
    
    async def extract_faces(
        self,
        file_content: bytes,
        document_name: str,
        min_confidence: float = 0.3,
        extract_all_faces: bool = True
    ) -> FaceExtractionResponse:
        """
        Extract faces from a document using agent execution.
        
        Args:
            file_content: Binary content of the document
            document_name: Name of the document
            min_confidence: Minimum confidence threshold for face detection
            extract_all_faces: Whether to extract all faces or just the first
            
        Returns:
            FaceExtractionResponse with extracted faces
        """
        start_time = time.time()
        
        try:
            # Set context for tool execution
            self.set_context({
                "file_content": file_content,
                "document_name": document_name,
                "min_confidence": min_confidence,
                "extract_all_faces": extract_all_faces
            })
            
            # Build task prompt
            task_prompt = self._build_task_prompt(document_name, min_confidence, extract_all_faces)
            
            # Execute task using agent
            result = await self.execute_task(task_prompt)
            
            if not result.get("success"):
                # Fallback to direct tool execution
                return await self._execute_fallback(start_time)
            
            # Check if faces are available in context
            ctx = result.get("context", {})
            if 'document_id' in ctx and 'extracted_faces' in ctx:
                return self._build_response_from_context(ctx, start_time)
            
            # If no results in context, try fallback
            return await self._execute_fallback(start_time)
            
        except Exception as e:
            logger.error(f"Error in extract_faces: {e}", exc_info=True)
            return FaceExtractionResponse(
                document_id="",
                faces_detected=0,
                faces_extracted=[],
                processing_time=time.time() - start_time,
                status=f"error: {str(e)}"
            )
        finally:
            self.clear_context()
    
    def _build_task_prompt(
        self,
        document_name: str,
        min_confidence: float,
        extract_all_faces: bool
    ) -> str:
        """Build task prompt for agent execution."""
        return f"""Extract faces from the uploaded document image and return the extracted face images.

Task parameters:
- Document name: {document_name}
- Minimum confidence threshold: {min_confidence}
- Extract all faces: {extract_all_faces}

You MUST execute ALL of these steps in order:
1. Call validate_document(document_name="{document_name}") - Validate the image format
2. Call upload_document(document_name="{document_name}") - Generate document_id for tracking
3. Call detect_faces(min_confidence={min_confidence}) - Detect faces with confidence >= {min_confidence}
4. Call extract_face_images() - Extract face crops from the detected faces. This is REQUIRED - you must call this tool after detecting faces.

IMPORTANT: The file content is available in the tool execution context - you do not need to pass it as a parameter.
After calling extract_face_images, the extracted faces will be available in the context.
You must complete ALL 4 steps to finish the task."""
    
    def _build_response_from_context(self, ctx: Dict[str, Any], start_time: float) -> FaceExtractionResponse:
        """Build FaceExtractionResponse from context."""
        extracted_faces_raw = ctx.get('extracted_faces', [])
        
        # Convert to ExtractedFace models, ensuring image_data is bytes
        extracted_faces = []
        for face_dict in extracted_faces_raw:
            # Ensure image_data is bytes, not base64 string
            if isinstance(face_dict.get("image_data"), str):
                face_dict["image_data"] = base64.b64decode(face_dict["image_data"])
            elif not isinstance(face_dict.get("image_data"), bytes):
                continue
            extracted_faces.append(ExtractedFace(**face_dict))
        
        logger.debug(f"Agent execution successful - found {len(extracted_faces)} faces in context")
        return FaceExtractionResponse(
            document_id=ctx.get('document_id', ''),
            faces_detected=len(extracted_faces),
            faces_extracted=extracted_faces,
            processing_time=time.time() - start_time,
            status="success"
        )
    
    async def _execute_fallback(self, start_time: float) -> FaceExtractionResponse:
        """Execute fallback workflow using tools directly."""
        try:
            ctx = self.get_context()
            
            # Validate document
            validation_result = await self.tools.validate_document(ctx["document_name"])
            if not validation_result.get("valid"):
                return FaceExtractionResponse(
                    document_id="",
                    faces_detected=0,
                    faces_extracted=[],
                    processing_time=time.time() - start_time,
                    status=f"error: invalid image - {validation_result.get('error', 'unknown error')}"
                )
            
            # Upload document
            document_info = await self.tools.upload_document(ctx["document_name"])
            document_id = document_info["document_id"]
            
            # Detect faces
            face_detections = await self.tools.detect_faces(
                min_confidence=ctx["min_confidence"],
                max_results=None if ctx["extract_all_faces"] else 1
            )
            
            if not face_detections:
                return FaceExtractionResponse(
                    document_id=document_id,
                    faces_detected=0,
                    faces_extracted=[],
                    processing_time=time.time() - start_time,
                    status="success: no faces detected"
                )
            
            # Extract faces
            extracted_faces_dicts = await self.tools.extract_face_images(
                document_id=document_id
            )
            
            # Convert to Pydantic models
            extracted_faces = []
            for face_dict in extracted_faces_dicts:
                if isinstance(face_dict.get("image_data"), str):
                    face_dict["image_data"] = base64.b64decode(face_dict["image_data"])
                extracted_faces.append(ExtractedFace(**face_dict))
            
            return FaceExtractionResponse(
                document_id=document_id,
                faces_detected=len(extracted_faces),
                faces_extracted=extracted_faces,
                processing_time=time.time() - start_time,
                status="success"
            )
        except Exception as e:
            logger.error(f"Fallback workflow error: {e}", exc_info=True)
            return FaceExtractionResponse(
                document_id="",
                faces_detected=0,
                faces_extracted=[],
                processing_time=time.time() - start_time,
                status=f"error: {str(e)}"
            )
