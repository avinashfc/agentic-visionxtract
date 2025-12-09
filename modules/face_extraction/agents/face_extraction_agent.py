"""
ADK-based agent for face extraction from documents.

Configuration is kept in modules/face_extraction/config.yaml to make the agent reusable and
easy to tune without code changes.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from google.adk import Agent

from modules.face_extraction.tools.face_extraction_tools import FaceExtractionTools

logger = logging.getLogger(__name__)


class FaceExtractionAgent:
    """ADK-based agent for extracting faces from uploaded documents."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize ADK face extraction agent.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name to use (overrides config if provided)
            config_path: Optional path to YAML config. Defaults to modules/face_extraction/config.yaml
        """
        self.config = self._load_config(config_path)

        self.api_key = api_key
        model_from_config = self.config.get("agent", {}).get("model")
        if not model_name and not model_from_config:
            raise ValueError("model_name must be provided either as parameter or in config.yaml (agent.model)")
        self.model_name = model_name or model_from_config
        self.app_name = self.config.get("agent", {}).get("app_name", "face_extraction_app")
        self.agent_name = self.config.get("agent", {}).get("name", "face_extraction_agent")
        self.agent_description = self.config.get(
            "agent", {}
        ).get("description", "Agent for extracting faces from uploaded documents using Vision API")
        self.task_config = self.config.get("task", {})
        
        self.tools = FaceExtractionTools(api_key=api_key)
        
        self.agent = self._create_agent()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load agent configuration from YAML.
        """
        default_path = Path(__file__).parent.parent / "config.yaml"
        path = Path(config_path) if config_path else default_path
        if not path.exists():
            logger.warning(f"Face extraction config not found at {path}, using defaults")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load face extraction config from {path}: {e}")
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
        min_confidence: float,
        extract_all_faces: bool
    ) -> str:
        """Build task prompt for agent execution from config."""
        # Build ordered steps from config; fallback to defaults
        steps = self.task_config.get("steps") or [
            "validate_document",
            "upload_document",
            "detect_faces",
            "extract_face_images",
        ]
        ordered = []
        for idx, step in enumerate(steps, start=1):
            if step == "validate_document":
                ordered.append(f"{idx}. Call validate_document(document_name=\"{document_name}\") - Validate the image format")
            elif step == "upload_document":
                ordered.append(f"{idx}. Call upload_document(document_name=\"{document_name}\") - Generate document_id for tracking")
            elif step == "detect_faces":
                ordered.append(f"{idx}. Call detect_faces(min_confidence={min_confidence}) - Detect faces with confidence >= {min_confidence}")
            elif step == "extract_face_images":
                ordered.append(f"{idx}. Call extract_face_images() - Extract face crops from the detected faces. This is REQUIRED - you must call this tool after detecting faces.")
            else:
                # Generic fallback
                ordered.append(f"{idx}. Call {step}()")

        ordered_steps = "\n".join(ordered)

        template = self.task_config.get("prompt_template")
        if template:
            return template.format(
                document_name=document_name,
                min_confidence=min_confidence,
                extract_all_faces=extract_all_faces,
                ordered_steps=ordered_steps,
            )

        # Fallback template if config missing
        fallback = f"""Extract faces from the uploaded document image and return the extracted face images.

Task parameters:
- Document name: {document_name}
- Minimum confidence threshold: {min_confidence}
- Extract all faces: {extract_all_faces}

You MUST execute ALL of these steps in order:
{ordered_steps}

IMPORTANT: The file content is available in the tool execution context - you do not need to pass it as a parameter.
After calling extract_face_images, the extracted faces will be available in the context.
You must complete ALL 4 steps to finish the task."""
        return fallback
