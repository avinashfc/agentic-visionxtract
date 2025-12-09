"""
Main FastAPI application entrypoint.
Automatically discovers and loads all available modules.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.module_registry import get_registry

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
    
    # Set GOOGLE_APPLICATION_CREDENTIALS if specified in .env
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and not os.path.exists(creds_path):
        # If it's a relative path, make it relative to project root
        abs_creds_path = Path(__file__).parent.parent / creds_path
        if abs_creds_path.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(abs_creds_path)
            print(f"Set GOOGLE_APPLICATION_CREDENTIALS to {abs_creds_path}")
else:
    print(f"Warning: .env file not found at {env_path}")

app = FastAPI(
    title="Agentic VisionXtract",
    description="Modular AI agent system with auto-discovered modules",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auto-discover and load modules
# ENABLE_MODULES can be: "all", "face_extraction", "ocr", "document_extraction", or comma-separated
enabled_modules_env = os.getenv("ENABLE_MODULES", "all").lower()
enabled_modules = {m.strip() for m in enabled_modules_env.split(",") if m.strip()}

print("\n" + "="*60)
print("Discovering modules...")
print("="*60)

registry = get_registry()
discovered_modules = registry.discover_modules(enabled_modules=enabled_modules)

# Register all discovered modules
for module_name, module_info in discovered_modules.items():
    app.include_router(
        module_info.router,
        prefix=module_info.prefix,
        tags=module_info.tags
    )
    print(f"✓ Registered module: {module_name} at {module_info.prefix}")

print("="*60)
print(f"Total modules loaded: {len(discovered_modules)}")
print("="*60 + "\n")


@app.get("/")
async def root():
    """Root endpoint."""
    module_names = list(discovered_modules.keys())
    return {
        "message": "Agentic VisionXtract API",
        "status": "running",
        "modules": module_names,
        "total_modules": len(module_names)
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "modules": list(discovered_modules.keys())}


@app.get("/modules")
async def list_modules():
    """List all available modules."""
    return {
        "modules": [
            {
                "name": info.name,
                "prefix": info.prefix,
                "tags": info.tags,
                "description": info.description,
                "version": info.version,
                "enabled": info.enabled
            }
            for info in discovered_modules.values()
        ]
    }

