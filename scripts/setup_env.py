"""
Setup script for environment configuration.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

def setup_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print(f".env file not found at {env_path}")
        print("Creating .env.example file...")
        create_env_example()

def create_env_example():
    """Create .env.example file with template variables."""
    env_example = Path(__file__).parent.parent / ".env.example"
    example_content = """# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
"""
    with open(env_example, "w") as f:
        f.write(example_content)
    print(f"Created .env.example at {env_example}")

if __name__ == "__main__":
    setup_env()

