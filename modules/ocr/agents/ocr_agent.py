"""
ADK-based agent for OCR and key-value extraction.

Configuration is kept in modules/ocr/config.yaml to make the agent reusable and
easy to tune without code changes.
"""
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from google.adk import Agent
from google.genai import types

from modules.ocr.tools.ocr_tools import OCRTools
from modules.ocr.models.ocr import KeyValueResponse

logger = logging.getLogger(__name__)


class OCRAgent:
    """ADK-based agent for OCR and key-value extraction from documents."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize ADK OCR agent.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name to use (overrides config if provided)
            config_path: Optional path to YAML config. Defaults to modules/ocr/config.yaml
        """
        self.config = self._load_config(config_path)

        self.api_key = api_key
        model_from_config = self.config.get("agent", {}).get("model")
        if not model_name and not model_from_config:
            raise ValueError("model_name must be provided either as parameter or in config.yaml (agent.model)")
        self.model_name = model_name or model_from_config
        self.app_name = self.config.get("agent", {}).get("app_name", "ocr_app")
        self.agent_name = self.config.get("agent", {}).get("name", "ocr_agent")
        self.agent_description = self.config.get(
            "agent", {}
        ).get("description", "Agent for extracting key-value pairs from documents using OCR and LLM")
        self.task_config = self.config.get("task", {})
        
        self.tools = OCRTools(api_key=api_key, model_name=self.model_name)
        
        self.agent = self._create_agent()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load agent configuration from YAML.
        """
        default_path = Path(__file__).parent.parent / "config.yaml"
        path = Path(config_path) if config_path else default_path
        if not path.exists():
            logger.warning(f"OCR config not found at {path}, using defaults")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load OCR config from {path}: {e}")
            return {}
    
    def _create_agent(self) -> Agent:
        """Create ADK Agent instance with tools."""
        agent_tools = self.tools.get_tools()
        
        agent_config = {
            "name": self.agent_name,
            "description": self.agent_description,
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
    
    def build_task_prompt(
        self,
        document_name: str,
        language_hints: Optional[list[str]],
        extraction_prompt: Optional[str]
    ) -> str:
        """Build task prompt for agent execution from config."""
        language_hints_str = json.dumps(language_hints) if language_hints else "null"

        # Build ordered steps from config; fallback to defaults
        steps = self.task_config.get("steps") or [
            "validate_document",
            "upload_document",
            "extract_text",
            "extract_key_value_pairs",
        ]
        ordered = []
        for idx, step in enumerate(steps, start=1):
            if step == "validate_document":
                ordered.append(f"{idx}. Call validate_document(document_name=\"{document_name}\") - Validate the image format")
            elif step == "upload_document":
                ordered.append(f"{idx}. Call upload_document(document_name=\"{document_name}\") - Generate a document_id for tracking")
            elif step == "extract_text":
                ordered.append(f"{idx}. Call extract_text(language_hints_json={language_hints_str}) - Extract all text from the image using OCR")
            elif step == "extract_key_value_pairs":
                if extraction_prompt:
                    ordered.append(f"{idx}. Call extract_key_value_pairs(extraction_prompt={json.dumps(extraction_prompt)}) - Extract key-value pairs using custom prompt")
                else:
                    ordered.append(f"{idx}. Call extract_key_value_pairs() - Extract key-value pairs from the OCR text")
            else:
                # Generic fallback
                ordered.append(f"{idx}. Call {step}()")

        ordered_steps = "\n".join(ordered)

        template = self.task_config.get("prompt_template")
        if template:
            return template.format(
                document_name=document_name,
                language_hints_json=language_hints_str,
                ordered_steps=ordered_steps,
            )

        # Fallback template if config missing
        fallback = f"""Extract key-value pairs from the uploaded document image.

                Task parameters:
                - Document name: {document_name}
                - Language hints: {language_hints_str}

                You MUST execute ALL of these steps in order:
                {ordered_steps}

                IMPORTANT: The file content is available in the tool execution context.
                After calling extract_text, the extracted text will be available in the context.
                After calling extract_key_value_pairs, the key-value pairs will be available in the context.
                You must complete ALL steps to finish the task."""
        return fallback
    
    # Response building moved to helpers/response_builder.py