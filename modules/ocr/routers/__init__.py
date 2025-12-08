"""
Routers for OCR module.
"""
from fastapi import APIRouter
from modules.ocr.routers.ocr import router as ocr_router

# Create main router for OCR module
router = APIRouter()

# Include sub-routers (no prefix since this is the main router)
router.include_router(
    ocr_router,
    tags=["ocr"]
)

