"""
ADK-based agent for LLM judging and evaluation.

Configuration is kept in modules/llm_judge/config.yaml to make the agent reusable and
easy to tune without code changes.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from google.adk import Agent

from modules.llm_judge.tools.judge_tools import JudgeTools

logger = logging.getLogger(__name__)


class JudgeAgent:
    """ADK-based agent for LLM judging and evaluation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize ADK judge agent.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name to use (overrides config if provided)
            config_path: Optional path to YAML config. Defaults to modules/llm_judge/config.yaml
        """
        self.config = self._load_config(config_path)
        
        self.api_key = api_key
        model_from_config = self.config.get("agent", {}).get("model")
        if not model_name and not model_from_config:
            raise ValueError("model_name must be provided either as parameter or in config.yaml (agent.model)")
        self.model_name = model_name or model_from_config
        self.app_name = self.config.get("agent", {}).get("app_name", "agents")
        self.agent_name = self.config.get("agent", {}).get("name", "judge_agent")
        self.agent_description = self.config.get(
            "agent", {}
        ).get("description", "Agent for evaluating and judging content using LLM with structured criteria")
        
        self.tools = JudgeTools(api_key=api_key, model_name=self.model_name)
        
        self.agent = self._create_agent()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load agent configuration from YAML.
        """
        default_path = Path(__file__).parent.parent / "config.yaml"
        path = Path(config_path) if config_path else default_path
        if not path.exists():
            logger.warning(f"Judge config not found at {path}, using defaults")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load judge config from {path}: {e}")
            return {}
    
    def _create_agent(self) -> Agent:
        """Create ADK Agent instance with judge tools."""
        agent_tools = self.tools.get_tools()
        
        agent_config = {
            "name": self.agent_name,
            "description": self.agent_description,
            "model": self.model_name,
            "tools": agent_tools,
        }
        
        agent = Agent(**agent_config)
        return agent
