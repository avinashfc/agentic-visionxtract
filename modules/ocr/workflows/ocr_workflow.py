import logging
import time
import json
import uuid
import io
from typing import Optional, List, Dict, Any
from PIL import Image
from google.adk import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from modules.ocr.agents.ocr_agent import OCRAgent
from modules.ocr.models.ocr import KeyValueResponse
from modules.ocr.helpers import (
    build_kv_response_from_context,
    evaluate_with_judge as evaluate_with_judge_fn,
)
from shared.tools.pdf_converter import PDFConverter

logger = logging.getLogger(__name__)


class OCRWorkflow:
    """ADK-based workflow orchestrator for OCR and key-value extraction pipeline."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize OCR workflow using ADK agentic flow.
        
        Args:
            api_key: Google API key
            model_name: Gemini model name (overrides config if provided)
        """
        self.agent = OCRAgent(
            api_key=api_key,
            model_name=model_name
        )
    
        self.runner = Runner(
            app_name=self.agent.app_name,
            agent=self.agent.agent,
            session_service=InMemorySessionService()
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
        start_time = time.time()
        try:
            # Check if input is a PDF and convert to image if needed
            processed_content = file_content
            if PDFConverter.is_pdf(file_content, document_name):
                logger.info(f"Detected PDF file: {document_name}, converting first page to image")
                image_bytes = PDFConverter.convert_pdf_to_image(file_content, page_index=0)
                if not image_bytes:
                    return self._error_response(
                        start_time=start_time,
                        message="error: failed to convert PDF to image. Ensure pdf2image is installed (pip install pdf2image) and poppler is available (brew install poppler on macOS)"
                    )
                processed_content = image_bytes
                logger.info(f"Successfully converted PDF to image ({len(processed_content)} bytes)")
            
            # Validate that we have a valid image
            try:
                Image.open(io.BytesIO(processed_content))
            except Exception as e:
                return self._error_response(
                    start_time=start_time,
                    message=f"error: invalid image input - {str(e)}"
                )

            # Set context for tool execution (use processed content, which may be converted from PDF)
            self.agent.set_context({
                "file_content": processed_content,
                "document_name": document_name,
                "language_hints": language_hints or []
            })

            task_prompt = self.agent.build_task_prompt(document_name, language_hints, extraction_prompt)

            result = await self._run_agent_task(
                task_prompt=task_prompt,
                user_id="ocr_user"
            )

            if not result.get("success"):
                return self._error_response(
                    start_time=start_time,
                    message=f"error: {result.get('error', 'agent execution failed')}"
                )

            ctx = result.get("context", {})
            # Check if we have at least document_id and either full_text or key_value_pairs
            if 'document_id' in ctx and ('full_text' in ctx or 'key_value_pairs' in ctx):
                response = build_kv_response_from_context(ctx, start_time)
                if evaluate_with_judge:
                    return await evaluate_with_judge_fn(
                        response,
                        document_name=document_name,
                        language_hints=language_hints,
                        judge_criteria=judge_criteria,
                        judge_task_description=judge_task_description,
                    )
                return response

            # If agent completed but context lacks expected outputs, return error
            return self._error_response(
                start_time=start_time,
                message="error: no document_id or extraction results in context"
            )

        finally:
            self.agent.clear_context()

    async def _run_agent_task(
        self,
        task_prompt: str,
        user_id: str = "ocr_user"
    ) -> Dict[str, Any]:
        """Run the ADK agent task via the workflow-owned runner."""
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
                "context": self.agent.get_context()
            }
        except Exception as e:
            logger.error(f"Agent execution error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "context": self.agent.get_context()
            }

    def _error_response(self, start_time: float, message: str) -> KeyValueResponse:
        """Build a standardized error response."""
        return KeyValueResponse(
            document_id="",
            key_value_pairs=[],
            raw_text="",
            processing_time=time.time() - start_time,
            status=message,
        )