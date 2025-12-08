"""
Module Registry and Discovery System

Automatically discovers and registers available modules from the modules directory.
This enables scalable module addition without modifying main application code.
"""
import os
import importlib
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from fastapi import APIRouter


@dataclass
class ModuleInfo:
    """Metadata about a discovered module."""
    name: str
    router: APIRouter
    prefix: str
    tags: List[str]
    enabled: bool = True
    description: Optional[str] = None
    version: Optional[str] = None


class ModuleRegistry:
    """Registry for auto-discovering and managing modules."""
    
    def __init__(self, modules_base_path: Optional[Path] = None):
        """
        Initialize module registry.
        
        Args:
            modules_base_path: Path to modules directory. Defaults to project modules/.
        """
        if modules_base_path is None:
            # Default to modules/ directory relative to this file
            # core/module_registry.py -> core/ -> project_root/ -> project_root/modules/
            project_root = Path(__file__).parent.parent
            modules_base_path = project_root / "modules"
        
        self.modules_base_path = modules_base_path
        self._modules: Dict[str, ModuleInfo] = {}
        self._discovered = False
    
    def discover_modules(self, enabled_modules: Optional[Set[str]] = None) -> Dict[str, ModuleInfo]:
        """
        Discover all available modules with routers.
        
        Args:
            enabled_modules: Set of module names to enable. If None, enables all.
                           Can contain "all" to enable everything.
        
        Returns:
            Dictionary mapping module names to ModuleInfo objects.
        """
        if self._discovered:
            return self._modules
        
        if not self.modules_base_path.exists():
            print(f"Warning: Modules directory not found at {self.modules_base_path}")
            return {}
        
        enabled_set = enabled_modules or {"all"}
        enable_all = "all" in enabled_set
        
        # Scan modules directory
        for module_dir in self.modules_base_path.iterdir():
            if not module_dir.is_dir() or module_dir.name.startswith("_"):
                continue
            
            module_name = module_dir.name
            
            # Check if module should be enabled
            if not enable_all and module_name not in enabled_set:
                continue
            
            # Check if module has routers directory
            routers_dir = module_dir / "routers"
            if not routers_dir.exists() or not (routers_dir / "__init__.py").exists():
                continue
            
            try:
                # Try to import the router
                module_import_path = f"modules.{module_name}.routers"
                router_module = importlib.import_module(module_import_path)
                
                if not hasattr(router_module, "router"):
                    print(f"Warning: Module {module_name} has routers/ but no 'router' export")
                    continue
                
                router = router_module.router
                
                # Load metadata from config.yaml if it exists
                config_path = module_dir / "config.yaml"
                metadata = self._load_module_metadata(config_path, module_name)
                
                # Generate API prefix (convert snake_case to kebab-case)
                prefix = self._generate_prefix(module_name)
                
                # Generate tags
                tags = metadata.get("tags", [self._generate_tag(module_name)])
                
                module_info = ModuleInfo(
                    name=module_name,
                    router=router,
                    prefix=f"/api/{prefix}",
                    tags=tags,
                    enabled=True,
                    description=metadata.get("description"),
                    version=metadata.get("version")
                )
                
                self._modules[module_name] = module_info
                print(f"✓ Discovered module: {module_name} (prefix: {module_info.prefix})")
                
            except ImportError as e:
                print(f"Warning: Could not import router for module {module_name}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error discovering module {module_name}: {e}")
                continue
        
        self._discovered = True
        return self._modules
    
    def _load_module_metadata(self, config_path: Path, module_name: str) -> Dict:
        """Load module metadata from config.yaml if available."""
        metadata = {}
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and "module" in config:
                        module_config = config["module"]
                        metadata = {
                            "description": module_config.get("description"),
                            "version": module_config.get("version"),
                            "tags": [module_config.get("name", module_name)]
                        }
            except Exception as e:
                print(f"Warning: Could not load config.yaml for {module_name}: {e}")
        
        return metadata
    
    def _generate_prefix(self, module_name: str) -> str:
        """Generate API prefix from module name (snake_case -> kebab-case)."""
        return module_name.replace("_", "-")
    
    def _generate_tag(self, module_name: str) -> str:
        """Generate tag from module name."""
        return module_name.replace("_", "-")
    
    def get_module(self, name: str) -> Optional[ModuleInfo]:
        """Get module info by name."""
        if not self._discovered:
            self.discover_modules()
        return self._modules.get(name)
    
    def get_all_modules(self) -> Dict[str, ModuleInfo]:
        """Get all discovered modules."""
        if not self._discovered:
            self.discover_modules()
        return self._modules
    
    def list_module_names(self) -> List[str]:
        """List names of all discovered modules."""
        return list(self.get_all_modules().keys())


# Global registry instance
_registry: Optional[ModuleRegistry] = None


def get_registry() -> ModuleRegistry:
    """Get or create global module registry instance."""
    global _registry
    if _registry is None:
        _registry = ModuleRegistry()
    return _registry

