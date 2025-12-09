"""
Module Client for Agent-to-Agent (A2A) Communication

Provides a unified interface for inter-module communication that supports:
- In-process mode: Direct Python function calls (unified deployment)
- HTTP mode: REST API calls (distributed/microservices deployment)
"""
import os
import importlib
import httpx
from typing import Optional, Dict, Any, List
from enum import Enum
import logging
from core.module_registry import get_registry

logger = logging.getLogger(__name__)


class CommunicationMode(Enum):
    """Communication mode for module interaction."""
    IN_PROCESS = "in_process"  # Direct Python calls
    HTTP = "http"  # REST API calls
    AUTO = "auto"  # Auto-detect based on configuration


class ModuleClient:
    """
    Client for Agent-to-Agent communication between modules.
    
    Supports both in-process (direct import) and HTTP (REST API) modes.
    Automatically detects the appropriate mode based on configuration.
    """
    
    def __init__(
        self,
        module_name: str,
        mode: CommunicationMode = CommunicationMode.AUTO,
        base_url: Optional[str] = None
    ):
        """
        Initialize module client.
        
        Args:
            module_name: Name of the target module (e.g., "llm_judge")
            mode: Communication mode (AUTO, IN_PROCESS, HTTP)
            base_url: Base URL for HTTP mode (e.g., "http://localhost:8002")
                     If None and mode is HTTP, will use MODULE_{NAME}_URL env var
        """
        self.module_name = module_name
        self.mode = mode
        self.base_url = base_url
        
        # Determine actual mode
        self._actual_mode = self._determine_mode()
        
        # Initialize based on mode
        if self._actual_mode == CommunicationMode.IN_PROCESS:
            self._workflow = self._load_in_process_workflow()
            self._http_client = None
        else:
            self._workflow = None
            self._http_client = httpx.AsyncClient(timeout=30.0)
            if not self.base_url:
                env_var = f"MODULE_{module_name.upper()}_URL"
                self.base_url = os.getenv(env_var)
                if not self.base_url:
                    # Default to same host with module-specific port
                    default_ports = {
                        "llm_judge": "8003",
                        "ocr": "8002",
                        "face_extraction": "8001"
                    }
                    port = default_ports.get(module_name, "8000")
                    self.base_url = f"http://localhost:{port}"
        
        logger.info(
            f"ModuleClient for '{module_name}' initialized in {self._actual_mode.value} mode"
            + (f" (base_url: {self.base_url})" if self._actual_mode == CommunicationMode.HTTP else "")
        )
    
    def _determine_mode(self) -> CommunicationMode:
        """Determine the actual communication mode to use."""
        if self.mode == CommunicationMode.AUTO:
            # Check if module URL is configured (indicates distributed deployment)
            env_var = f"MODULE_{self.module_name.upper()}_URL"
            if os.getenv(env_var):
                return CommunicationMode.HTTP
            
            # Check if we're in unified deployment (all modules in same process)
            # Try to import the module - if successful, use in-process
            try:
                self._load_in_process_workflow()
                return CommunicationMode.IN_PROCESS
            except (ImportError, AttributeError):
                # Module not available in-process, use HTTP
                return CommunicationMode.HTTP
        else:
            return self.mode
    
    def _load_in_process_workflow(self):
        """Load workflow class for in-process communication using module registry."""
        # Use module registry to discover modules
        registry = get_registry()
        discovered_modules = registry.discover_modules()
        
        if self.module_name not in discovered_modules:
            raise ImportError(f"Module '{self.module_name}' not found in discovered modules")
        
        # Try to import workflow from workflows module
        # Strategy: First check workflows/__init__.py for exported workflow class
        # Then try direct import from workflow file
        
        workflows_module_path = f"modules.{self.module_name}.workflows"
        
        try:
            # First, try to import from workflows/__init__.py
            workflows_module = importlib.import_module(workflows_module_path)
            
            # Look for workflow class in __all__ (preferred method)
            if hasattr(workflows_module, "__all__"):
                for export_name in workflows_module.__all__:
                    if "Workflow" in export_name:
                        workflow_class = getattr(workflows_module, export_name)
                        break
                else:
                    raise AttributeError(f"No workflow class found in {workflows_module_path}.__all__")
            else:
                # No __all__, try to find workflow class by common patterns
                # Try common workflow file names
                workflow_files = [
                    f"{self.module_name}_workflow",
                    "judge_workflow" if self.module_name == "llm_judge" else None,
                    f"{self.module_name.replace('_', '')}_workflow"
                ]
                workflow_files = [f for f in workflow_files if f]
                
                workflow_class = None
                for workflow_file in workflow_files:
                    try:
                        workflow_module = importlib.import_module(f"{workflows_module_path}.{workflow_file}")
                        # Look for class ending in "Workflow"
                        for attr_name in dir(workflow_module):
                            if attr_name.endswith("Workflow") and not attr_name.startswith("_"):
                                workflow_class = getattr(workflow_module, attr_name)
                                break
                        if workflow_class:
                            break
                    except ImportError:
                        continue
                
                if not workflow_class:
                    raise ImportError(f"Could not find workflow class in {workflows_module_path}")
        except (ImportError, AttributeError) as e:
            logger.error(f"Could not import workflow from {workflows_module_path}: {e}")
            raise ImportError(f"Module '{self.module_name}' workflow not found: {e}")
        
        # Initialize workflow with API key
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")
        
        return workflow_class(api_key=api_key, model_name=model_name)
    
    async def call(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generic method to call a module endpoint.
        
        Args:
            method: HTTP method or workflow method name
            endpoint: API endpoint path (for HTTP) or method name (for in-process)
            payload: Request payload
            params: Query parameters (HTTP only)
            
        Returns:
            Response data
        """
        if self._actual_mode == CommunicationMode.IN_PROCESS:
            return await self._call_in_process(endpoint, payload)
        else:
            return await self._call_http(method, endpoint, payload, params)
    
    async def _call_in_process(self, method_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call module method in-process."""
        if not hasattr(self._workflow, method_name):
            raise AttributeError(f"Method '{method_name}' not found in {self.module_name} workflow")
        
        method = getattr(self._workflow, method_name)
        
        # Convert payload dict to method arguments
        if payload:
            result = await method(**payload)
        else:
            result = await method()
        
        # Convert Pydantic models to dicts
        if hasattr(result, 'model_dump'):
            return result.model_dump(mode='json')
        elif hasattr(result, 'dict'):
            return result.dict()
        else:
            return result
    
    async def _call_http(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call module via HTTP."""
        url = f"{self.base_url}/api/{self.module_name.replace('_', '-')}/{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = await self._http_client.get(url, params=params)
            elif method.upper() == "POST":
                response = await self._http_client.post(url, json=payload, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling {url}: {e}")
            raise
    
    # Convenience methods for common operations
    
    async def evaluate(
        self,
        content: str,
        reference: Optional[str] = None,
        criteria: Optional[List[Dict[str, Any]]] = None,
        task_description: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate content using judge module.
        
        Args:
            content: Content to evaluate
            reference: Optional reference content
            criteria: Optional evaluation criteria
            task_description: Optional task description
            context: Optional context
            
        Returns:
            JudgeResponse as dict
        """
        if self.module_name != "llm_judge":
            raise ValueError("evaluate() is only available for llm_judge module")
        
        if self._actual_mode == CommunicationMode.IN_PROCESS:
            result = await self._workflow.execute(
                content=content,
                reference=reference,
                criteria=criteria,
                task_description=task_description,
                context=context
            )
            # Convert Pydantic model to dict
            if hasattr(result, 'model_dump'):
                return result.model_dump(mode='json')
            elif hasattr(result, 'dict'):
                return result.dict()
            return result
        else:
            payload = {
                "content": content,
                "reference": reference,
                "criteria": criteria,
                "task_description": task_description,
                "context": context
            }
            return await self._call_http("POST", "evaluate", payload)
    
    async def compare(
        self,
        outputs: List[str],
        criteria: Optional[List[Dict[str, Any]]] = None,
        task_description: Optional[str] = None,
        rank: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple outputs using judge module.
        
        Args:
            outputs: List of outputs to compare
            criteria: Optional evaluation criteria
            task_description: Optional task description
            rank: Whether to rank outputs
            
        Returns:
            ComparisonResponse as dict
        """
        if self.module_name != "llm_judge":
            raise ValueError("compare() is only available for llm_judge module")
        
        if self._actual_mode == CommunicationMode.IN_PROCESS:
            result = await self._workflow.execute_comparison(
                outputs=outputs,
                criteria=criteria,
                task_description=task_description,
                rank=rank
            )
            # Convert Pydantic model to dict
            if hasattr(result, 'model_dump'):
                return result.model_dump(mode='json')
            elif hasattr(result, 'dict'):
                return result.dict()
            return result
        else:
            payload = {
                "outputs": outputs,
                "criteria": criteria,
                "task_description": task_description,
                "rank": rank
            }
            return await self._call_http("POST", "compare", payload)
    
    async def close(self):
        """Close HTTP client if in HTTP mode."""
        if self._http_client:
            await self._http_client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

