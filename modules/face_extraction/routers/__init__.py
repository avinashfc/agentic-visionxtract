"""
Routers for face extraction module.
"""
from fastapi import APIRouter
from modules.face_extraction.routers.face_extraction import router as face_extraction_router

# Create main router for face extraction module
router = APIRouter()

# Include sub-routers (no prefix since this is the main router)
router.include_router(
    face_extraction_router,
    tags=["face-extraction"]
)

