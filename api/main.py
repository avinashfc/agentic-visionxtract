"""
Main FastAPI application entrypoint.
Combines routes from all modules.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    description="Modular AI agent system with payments, fraud detection, face extraction, and OCR",
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

# Import and include routers from modules
# from modules.payments.routers import router as payments_router
# from modules.fraud_detection.routers import router as fraud_router
from modules.face_extraction.routers import router as face_extraction_router
from modules.ocr.routers import router as ocr_router
from modules.combined_extraction.routers import router as combined_extraction_router

# app.include_router(payments_router, prefix="/api/payments", tags=["payments"])
# app.include_router(fraud_router, prefix="/api/fraud", tags=["fraud-detection"])
app.include_router(face_extraction_router, prefix="/api/face-extraction", tags=["face-extraction"])
app.include_router(ocr_router, prefix="/api/ocr", tags=["ocr"])
app.include_router(combined_extraction_router, prefix="/api/combined-extraction", tags=["combined-extraction"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Agentic VisionXtract API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

