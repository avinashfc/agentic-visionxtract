"""
ADK-based agent for extracting key-value pairs from OCR results.
"""
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
    KeyValueResponse,
    KeyValuePair
)
import logging

logger = logging.getLogger(__name__)


class KeyValueAgent:
    """ADK-based agent for extracting key-value pairs from documents."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize ADK key-value extraction agent.
        
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
                logger.warning("Key-value extraction will not work without Gemini client.")
        
        self.agent = self._create_agent()
        
        self.runner = Runner(
            app_name="key_value_app",
            agent=self.agent,
            session_service=InMemorySessionService()
        )
    
    def _create_agent(self) -> Agent:
        agent_tools = self.tools.get_tools()
        
        agent_config = {
            "name": "key_value_agent",
            "description": "Agent for extracting key-value pairs from documents using OCR and LLM",
            "model": self.model_name,
            "tools": agent_tools,
        }
        
        agent = Agent(**agent_config)
        
        return agent
    
    async def extract_key_value_pairs(
        self,
        file_content: bytes,
        document_name: str,
        language_hints: Optional[list[str]] = None,
        extraction_prompt: Optional[str] = None
    ) -> KeyValueResponse:
        """
        Extract key-value pairs from a document.
        
        Args:
            file_content: Binary content of the document
            document_name: Name of the document
            language_hints: Optional list of language codes
            extraction_prompt: Optional custom prompt for key-value extraction
            
        Returns:
            KeyValueResponse with extracted key-value pairs
        """
        start_time = time.time()
        
        try:
            self.tools._context = {
                "file_content": file_content,
                "document_name": document_name,
                "language_hints": language_hints or []
            }
            
            language_hints_str = json.dumps(language_hints) if language_hints else "null"
            
            # Build task prompt
            prompt_parts = [
                f"Extract key-value pairs from the uploaded document image.",
                f"\nTask parameters:",
                f"- Document name: {document_name}",
                f"- Language hints: {language_hints_str}",
                f"\nYou MUST execute ALL of these steps in order:",
                f"1. Call validate_document(document_name=\"{document_name}\") - Validate the image format",
                f"2. Call upload_document(document_name=\"{document_name}\") - Generate a document_id for tracking.",
                f"3. Call extract_text(language_hints_json={language_hints_str}) - Extract all text from the image using OCR.",
            ]
            
            if extraction_prompt:
                prompt_parts.append(
                    f"4. Call extract_key_value_pairs(extraction_prompt={json.dumps(extraction_prompt)}) - Extract key-value pairs using custom prompt."
                )
            else:
                prompt_parts.append(
                    "4. Call extract_key_value_pairs() - Extract key-value pairs from the OCR text."
                )
            
            prompt_parts.extend([
                "\nIMPORTANT: The file content is available in the tool execution context.",
                "After calling extract_text, the extracted text will be available in the context.",
                "After calling extract_key_value_pairs, the key-value pairs will be available in the context.",
                "You must complete ALL 4 steps to finish the task."
            ])
            
            task_prompt = "\n".join(prompt_parts)
            
            return await self._execute_with_agent(task_prompt, start_time, document_name)
            
        except Exception as e:
            logger.error(f"Error in extract_key_value_pairs: {e}", exc_info=True)
            return KeyValueResponse(
                document_id="",
                key_value_pairs=[],
                raw_text="",
                processing_time=time.time() - start_time,
                status=f"error: {str(e)}"
            )
        finally:
            if hasattr(self.tools, '_context'):
                delattr(self.tools, '_context')
    
    async def _execute_with_agent(
        self,
        task_prompt: str,
        start_time: float,
        document_name: str
    ) -> KeyValueResponse:
        new_message = types.Content(parts=[types.Part(text=task_prompt)])
        
        user_id = "key_value_user"
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
                
                # Check if key-value pairs are available in context
                if hasattr(self.tools, '_context') and 'document_id' in self.tools._context:
                    ctx = self.tools._context
                    if 'key_value_pairs' in ctx:
                        key_value_pairs_raw = ctx.get('key_value_pairs', [])
                        full_text = ctx.get('full_text', '')
                        
                        # Convert to KeyValuePair objects
                        key_value_pairs = []
                        for kv in key_value_pairs_raw:
                            if isinstance(kv, dict):
                                key_value_pairs.append(KeyValuePair(
                                    key=kv.get('key', ''),
                                    value=kv.get('value', ''),
                                    confidence=kv.get('confidence')
                                ))
                        
                        logger.debug(f"Agent execution successful - extracted {len(key_value_pairs)} key-value pairs")
                        return KeyValueResponse(
                            document_id=ctx.get('document_id', ''),
                            key_value_pairs=key_value_pairs,
                            raw_text=full_text,
                            processing_time=time.time() - start_time,
                            status="success"
                        )
            
            logger.debug(f"Agent execution completed - {len(events)} events, {function_calls_executed} function calls executed")
        except Exception as e:
            logger.error(f"Agent execution error: {e}, using fallback workflow", exc_info=True)
            return await self._execute_fallback_workflow(start_time, document_name)
        
        # Check context one more time after all events
        if hasattr(self.tools, '_context') and 'document_id' in self.tools._context:
            ctx = self.tools._context
            if 'key_value_pairs' in ctx:
                key_value_pairs_raw = ctx.get('key_value_pairs', [])
                full_text = ctx.get('full_text', '')
                
                key_value_pairs = []
                for kv in key_value_pairs_raw:
                    if isinstance(kv, dict):
                        key_value_pairs.append(KeyValuePair(
                            key=kv.get('key', ''),
                            value=kv.get('value', ''),
                            confidence=kv.get('confidence')
                        ))
                
                return KeyValueResponse(
                    document_id=ctx.get('document_id', ''),
                    key_value_pairs=key_value_pairs,
                    raw_text=full_text,
                    processing_time=time.time() - start_time,
                    status="success"
                )
        
        return await self._execute_fallback_workflow(start_time, document_name)
    
    async def _execute_fallback_workflow(
        self,
        start_time: float,
        document_name: str
    ) -> KeyValueResponse:
        """Fallback workflow if agent execution fails."""
        ctx = self.tools._context
        
        # Validate document
        validation_result = await self.tools.validate_document(document_name)
        if not validation_result.get("valid"):
            return KeyValueResponse(
                document_id="",
                key_value_pairs=[],
                raw_text="",
                processing_time=time.time() - start_time,
                status=f"error: invalid image - {validation_result.get('error', 'unknown error')}"
            )
        
        # Upload document
        document_info = await self.tools.upload_document(document_name)
        document_id = document_info["document_id"]
        
        # Extract text
        language_hints = ctx.get('language_hints')
        language_hints_str = json.dumps(language_hints) if language_hints else None
        
        ocr_result = await self.tools.extract_text(language_hints_json=language_hints_str)
        
        if ocr_result.get("error"):
            return KeyValueResponse(
                document_id=document_id,
                key_value_pairs=[],
                raw_text="",
                processing_time=time.time() - start_time,
                status=f"error: {ocr_result.get('error')}"
            )
        
        # Extract key-value pairs
        kv_result = await self.tools.extract_key_value_pairs()
        
        if kv_result.get("error"):
            return KeyValueResponse(
                document_id=document_id,
                key_value_pairs=[],
                raw_text=ocr_result.get("full_text", ""),
                processing_time=time.time() - start_time,
                status=f"error: {kv_result.get('error')}"
            )
        
        # Convert to KeyValuePair objects
        key_value_pairs = []
        for kv in kv_result.get("key_value_pairs", []):
            if isinstance(kv, dict):
                key_value_pairs.append(KeyValuePair(
                    key=kv.get('key', ''),
                    value=kv.get('value', ''),
                    confidence=kv.get('confidence')
                ))
        
        return KeyValueResponse(
            document_id=document_id,
            key_value_pairs=key_value_pairs,
            raw_text=ocr_result.get("full_text", ""),
            processing_time=time.time() - start_time,
            status="success"
        )
    
    def get_agent(self) -> Agent:
        return self.agent


