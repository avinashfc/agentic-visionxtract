import time
import base64
import uuid
from typing import Optional, List
from google.adk import Agent, Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import Client, types
from modules.face_extraction.tools.face_extraction_tools import FaceExtractionTools
from modules.face_extraction.models.face_extraction import (
    FaceExtractionRequest,
    FaceExtractionResponse,
    ExtractedFace
)


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
        
        # Initialize tools
        self.tools = FaceExtractionTools(api_key=api_key)
        
        # Initialize genai client if API key is provided (for future use)
        self.genai_client = None
        if api_key:
            try:
                self.genai_client = Client(api_key=api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Gemini client: {e}")
                print("Face detection will still work using Vision API directly.")
        
        # Create ADK Agent
        self.agent = self._create_agent()
        
        # Create Runner with in-memory session service for agentic execution
        self.runner = Runner(
            app_name="face_extraction_app",
            agent=self.agent,
            session_service=InMemorySessionService()
        )
    
    def _create_agent(self) -> Agent:
        """
        Create ADK Agent instance with tools for agentic orchestration.
        
        Returns:
            ADK Agent instance
        """
        agent_tools = self.tools.get_tools()
        
        # Create agent configuration - Agent requires model, name, description, and tools
        agent_config = {
            "name": "face_extraction_agent",
            "description": "Agent for extracting faces from uploaded documents using Vision API",
            "model": self.model_name,  # Required: Agent needs a model to execute
            "tools": agent_tools,
        }
        
        # Create agent - it will use tools autonomously based on task prompts
        # The agent orchestrates tool calls based on the task description provided
        agent = Agent(**agent_config)
        
        return agent
    
    async def process_document(
        self,
        file_content: bytes,
        document_name: str,
        min_confidence: float = 0.3,  # Lower default threshold
        extract_all_faces: bool = True
    ) -> FaceExtractionResponse:
        """
        Process document to extract faces using ADK agentic flow.
        The agent autonomously orchestrates tool calls.
        
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
            # Store context in tools for agent to access during tool execution
            self.tools._context = {
                "file_content": file_content,
                "document_name": document_name,
                "min_confidence": min_confidence,
                "extract_all_faces": extract_all_faces
            }
            
            # Create agentic task prompt - agent will orchestrate tool calls
            task_prompt = f"""Extract faces from the uploaded document image and return the extracted face images.

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
            
            # Use Runner to execute agent with task prompt
            try:
                return await self._execute_with_agent(task_prompt, start_time)
            except Exception as agent_error:
                # Fallback to direct workflow if agent execution fails
                print(f"Agent execution error: {agent_error}, using fallback workflow")
                return await self._execute_agentic_workflow(start_time)
            
        except Exception as e:
            processing_time = time.time() - start_time
            return FaceExtractionResponse(
                document_id="",
                faces_detected=0,
                faces_extracted=[],
                processing_time=processing_time,
                status=f"error: {str(e)}"
            )
        finally:
            # Clean up context
            if hasattr(self.tools, '_context'):
                delattr(self.tools, '_context')
    
    async def _execute_with_agent(self, task_prompt: str, start_time: float) -> FaceExtractionResponse:
        """
        Execute agent using Runner with task prompt - truly agentic execution.
        The agent will autonomously decide which tools to call based on the task.
        
        Args:
            task_prompt: The task description for the agent
            start_time: Start time for processing
            
        Returns:
            FaceExtractionResponse
        """
        # Create Content object from task prompt
        new_message = types.Content(parts=[types.Part(text=task_prompt)])
        
        # Generate unique IDs for user and session
        user_id = "face_extraction_user"
        session_id = str(uuid.uuid4())
        
        # Create session first (required by Runner)
        session = await self.runner.session_service.create_session(
            app_name=self.runner.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        # Use Runner to execute agent with task prompt
        # Runner.run_async returns an async generator of events
        # The Runner automatically executes function calls from the agent
        events = []
        last_model_response = None
        function_calls_executed = 0
        try:
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message
            ):
                events.append(event)
                
                # Track function calls being executed
                if hasattr(event, 'get_function_calls'):
                    function_calls = event.get_function_calls()
                    if function_calls:
                        function_calls_executed += len(function_calls)
                        print(f"Debug: Agent executed {len(function_calls)} function call(s): {[fc.name for fc in function_calls]}")
                
                # Track function responses to see tool results
                if hasattr(event, 'get_function_responses'):
                    function_responses = event.get_function_responses()
                    if function_responses:
                        for fr in function_responses:
                            print(f"Debug: Function response from {fr.name}: {str(fr.response)[:100]}...")
                
                # Track the last model response event
                if event.author == 'model' and event.content:
                    last_model_response = event
                    # Check if model is saying the task is complete
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_lower = part.text.lower()
                                if 'complete' in text_lower or 'done' in text_lower or 'finished' in text_lower:
                                    print(f"Debug: Model indicates task complete: {part.text[:200]}")
                
                # Check if we have results in context after tool execution
                # Tools populate context during execution
                if hasattr(self.tools, '_context') and 'document_id' in self.tools._context:
                    # Agent has completed execution, extract results
                    ctx = self.tools._context
                    extracted_faces_raw = ctx.get('extracted_faces', [])
                    if extracted_faces_raw:
                        # Convert to ExtractedFace models, ensuring image_data is bytes
                        extracted_faces = []
                        for face_dict in extracted_faces_raw:
                            # Ensure image_data is bytes, not base64 string
                            if isinstance(face_dict.get("image_data"), str):
                                # It's already base64 encoded, decode it to bytes
                                face_dict["image_data"] = base64.b64decode(face_dict["image_data"])
                            elif not isinstance(face_dict.get("image_data"), bytes):
                                # Skip if invalid
                                continue
                            extracted_faces.append(ExtractedFace(**face_dict))
                        
                        if extracted_faces:
                            print(f"Debug: Agent execution successful - found {len(extracted_faces)} faces in context")
                            return FaceExtractionResponse(
                                document_id=ctx.get('document_id', ''),
                                faces_detected=len(extracted_faces),
                                faces_extracted=extracted_faces,
                                processing_time=time.time() - start_time,
                                status="success"
                            )
            
            print(f"Debug: Agent execution completed - {len(events)} events, {function_calls_executed} function calls executed")
        except Exception as e:
            # If agent execution fails, fall back to direct workflow
            print(f"Agent execution error: {e}, using fallback workflow")
            return await self._execute_agentic_workflow(start_time)
        
        # After all events are processed, check if we have results
        # The agent may have executed tools but results might be in context
        if hasattr(self.tools, '_context') and 'document_id' in self.tools._context:
            ctx = self.tools._context
            extracted_faces_raw = ctx.get('extracted_faces', [])
            if extracted_faces_raw:
                # Convert to ExtractedFace models, ensuring image_data is bytes
                extracted_faces = []
                for face_dict in extracted_faces_raw:
                    # Ensure image_data is bytes, not base64 string
                    if isinstance(face_dict.get("image_data"), str):
                        # It's already base64 encoded, decode it to bytes
                        face_dict["image_data"] = base64.b64decode(face_dict["image_data"])
                    elif not isinstance(face_dict.get("image_data"), bytes):
                        # Skip if invalid
                        continue
                    extracted_faces.append(ExtractedFace(**face_dict))
                
                if extracted_faces:
                    return FaceExtractionResponse(
                        document_id=ctx.get('document_id', ''),
                        faces_detected=len(extracted_faces),
                        faces_extracted=extracted_faces,
                        processing_time=time.time() - start_time,
                        status="success"
                    )
        
        # If agent execution didn't populate context, fall back to direct workflow
        # This happens if the agent didn't successfully execute the tools
        return await self._execute_agentic_workflow(start_time)
    
    async def _execute_agentic_workflow(self, start_time: float) -> FaceExtractionResponse:
        """
        Execute agentic workflow - agent orchestrates tool calls.
        The agent uses ADK FunctionTools registered with it to execute the task.
        Tool execution follows the agentic pattern where the agent decides the order.
        
        Args:
            start_time: Start time for processing
            
        Returns:
            FaceExtractionResponse
        """
        ctx = self.tools._context
        
        # Agent orchestrates: Step 1 - Validate
        validation_result = await self.tools.validate_document(ctx["document_name"])
        if not validation_result.get("valid"):
            return FaceExtractionResponse(
                document_id="",
                faces_detected=0,
                faces_extracted=[],
                processing_time=time.time() - start_time,
                status=f"error: invalid image - {validation_result.get('error', 'unknown error')}"
            )
        
        # Agent orchestrates: Step 2 - Upload
        document_info = await self.tools.upload_document(ctx["document_name"])
        document_id = document_info["document_id"]
        
        # Agent orchestrates: Step 3 - Detect faces
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
        
        # Agent orchestrates: Step 4 - Extract faces
        # Tools now get face_detections from context automatically
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
    
    async def extract_faces_from_request(
        self,
        request: FaceExtractionRequest,
        file_content: bytes
    ) -> FaceExtractionResponse:
        """
        Extract faces based on a request object.
        
        Args:
            request: FaceExtractionRequest object
            file_content: Binary content of the document
            
        Returns:
            FaceExtractionResponse with extracted faces
        """
        return await self.process_document(
            file_content=file_content,
            document_name=request.document_id,
            min_confidence=request.min_confidence,
            extract_all_faces=request.extract_all_faces
        )
    
    def get_agent(self) -> Agent:
        """Get the ADK Agent instance."""
        return self.agent

