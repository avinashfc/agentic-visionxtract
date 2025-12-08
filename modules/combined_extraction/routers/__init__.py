"""
Routers for combined extraction module.
"""
from fastapi import APIRouter
from modules.combined_extraction.routers.combined_extraction import router as combined_extraction_router

# Create main router for combined extraction module
router = APIRouter()

# Include sub-routers (no prefix since this is the main router)
router.include_router(
    combined_extraction_router,
    tags=["combined-extraction"]
)

