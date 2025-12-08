import json
import time
import uuid
from typing import Optional
from google.adk import Agent, Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import Client, types
from modules.ocr.tools.ocr_tools import OCRTools
from modules.ocr.models.ocr import (
    OCRRequest,
    OCRResponse,
    TextBlock
)
import logging

logger = logging.getLogger(__name__)


class OCRAgent:
    """ADK-based agent for extracting text from uploaded documents."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize ADK OCR agent.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        self.tools = OCRTools(api_key=api_key, model_name=model_name)
        
        self.genai_client = None
        if api_key:
            try:
                self.genai_client = Client(api_key=api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")
                logger.warning("OCR will still work using Vision API directly.")
        
        self.agent = self._create_agent()
        
        self.runner = Runner(
            app_name="ocr_app",
            agent=self.agent,
            session_service=InMemorySessionService()
        )
    
    def _create_agent(self) -> Agent:
        agent_tools = self.tools.get_tools()
        
        agent_config = {
            "name": "ocr_agent",
            "description": "Agent for extracting text from uploaded documents using Vision API OCR",
            "model": self.model_name,
            "tools": agent_tools,
        }
        
        agent = Agent(**agent_config)
        
        return agent
    
    async def process_document(
        self,
        file_content: bytes,
        document_name: str,
        language_hints: Optional[list[str]] = None
    ) -> OCRResponse:
        start_time = time.time()
        
        try:
            self.tools._context = {
                "file_content": file_content,
                "document_name": document_name,
                "language_hints": language_hints or []
            }
            
            language_hints_str = json.dumps(language_hints) if language_hints else "null"
            
            task_prompt = f"""Extract all text from the uploaded document image using OCR.

                           Task parameters:
                           - Document name: {document_name}
                           - Language hints: {language_hints_str}

                           You MUST execute ALL of these steps in order:
                           1. Call validate_document(document_name="{document_name}") - Validate the image format
                           2. Call upload_document(document_name="{document_name}") - Generate a document_id for tracking.
                           3. Call extract_text(language_hints_json={language_hints_str}) - Extract all text from the image using OCR.

                           IMPORTANT: The file content is available in the tool execution context - you do not need to pass it as a parameter.
                           After calling extract_text, the extracted text will be available in the context.
                           You must complete ALL 3 steps to finish the task."""
            
            return await self._execute_with_agent(task_prompt, start_time)
            
        except Exception as e:
            logger.error(f"Error in process_document: {e}", exc_info=True)
            return OCRResponse(
                document_id="",
                full_text="",
                text_blocks=[],
                languages_detected=[],
                processing_time=time.time() - start_time,
                status=f"error: {str(e)}"
            )
        finally:
            if hasattr(self.tools, '_context'):
                delattr(self.tools, '_context')
    
    async def _execute_with_agent(self, task_prompt: str, start_time: float) -> OCRResponse:
        new_message = types.Content(parts=[types.Part(text=task_prompt)])
        
        user_id = "ocr_user"
        session_id = str(uuid.uuid4())
        
        session = await self.runner.session_service.create_session(
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
                
                if hasattr(self.tools, '_context') and 'document_id' in self.tools._context:
                    ctx = self.tools._context
                    if 'full_text' in ctx:
                        text_blocks_raw = ctx.get('text_blocks', [])
                        text_blocks = [TextBlock(**block) for block in text_blocks_raw]
                        
                        logger.debug(f"Agent execution successful - extracted {len(text_blocks)} text blocks")
                        return OCRResponse(
                            document_id=ctx.get('document_id', ''),
                            full_text=ctx.get('full_text', ''),
                            text_blocks=text_blocks,
                            languages_detected=ctx.get('detected_languages', []),
                            processing_time=time.time() - start_time,
                            status="success"
                        )
            
            logger.debug(f"Agent execution completed - {len(events)} events, {function_calls_executed} function calls executed")
        except Exception as e:
            logger.error(f"Agent execution error: {e}, using fallback workflow", exc_info=True)
            return await self._execute_fallback_workflow(start_time)
        
        # Check context one more time after all events
        if hasattr(self.tools, '_context') and 'document_id' in self.tools._context:
            ctx = self.tools._context
            if 'full_text' in ctx:
                text_blocks_raw = ctx.get('text_blocks', [])
                text_blocks = [TextBlock(**block) for block in text_blocks_raw]
                
                return OCRResponse(
                    document_id=ctx.get('document_id', ''),
                    full_text=ctx.get('full_text', ''),
                    text_blocks=text_blocks,
                    languages_detected=ctx.get('detected_languages', []),
                    processing_time=time.time() - start_time,
                    status="success"
                )
        
        return await self._execute_fallback_workflow(start_time)
    
    async def _execute_fallback_workflow(self, start_time: float) -> OCRResponse:
        ctx = self.tools._context
        
        validation_result = await self.tools.validate_document(ctx["document_name"])
        if not validation_result.get("valid"):
            return OCRResponse(
                document_id="",
                full_text="",
                text_blocks=[],
                languages_detected=[],
                processing_time=time.time() - start_time,
                status=f"error: invalid image - {validation_result.get('error', 'unknown error')}"
            )
        
        document_info = await self.tools.upload_document(ctx["document_name"])
        document_id = document_info["document_id"]
        
        language_hints = ctx.get('language_hints')
        language_hints_str = json.dumps(language_hints) if language_hints else None
        
        ocr_result = await self.tools.extract_text(language_hints_json=language_hints_str)
        
        if ocr_result.get("error"):
            return OCRResponse(
                document_id=document_id,
                full_text="",
                text_blocks=[],
                languages_detected=[],
                processing_time=time.time() - start_time,
                status=f"error: {ocr_result.get('error')}"
            )
        
        text_blocks = [TextBlock(**block) for block in ocr_result.get("text_blocks", [])]
        
        return OCRResponse(
            document_id=document_id,
            full_text=ocr_result.get("full_text", ""),
            text_blocks=text_blocks,
            languages_detected=ocr_result.get("detected_languages", []),
            processing_time=time.time() - start_time,
            status="success"
        )
    
    async def extract_text_from_request(
        self,
        request: OCRRequest,
        file_content: bytes
    ) -> OCRResponse:
        return await self.process_document(
            file_content=file_content,
            document_name=request.document_id,
            language_hints=request.language_hints
        )
    
    def get_agent(self) -> Agent:
        return self.agent

