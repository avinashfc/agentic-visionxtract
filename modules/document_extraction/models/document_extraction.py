"""
Pydantic models for document extraction pipeline.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from modules.ocr.models.ocr import TextBlock, KeyValuePair


class DocumentExtractionResponse(BaseModel):
    """Model for document extraction response including faces, OCR, and key-value pairs."""
    document_id: str = Field(..., description="Document identifier")
    faces_detected: int = Field(..., description="Number of faces detected")
    faces_extracted: List[dict] = Field(default_factory=list, description="List of extracted faces")
    full_text: str = Field(..., description="Complete extracted text from document")
    text_blocks: List[TextBlock] = Field(default_factory=list, description="List of detected text blocks with positions")
    languages_detected: List[str] = Field(default_factory=list, description="Detected languages")
    key_value_pairs: List[KeyValuePair] = Field(default_factory=list, description="List of extracted key-value pairs")
    processing_time: float = Field(..., description="Total processing time in seconds")
    face_extraction_time: Optional[float] = Field(None, description="Face extraction processing time")
    ocr_time: Optional[float] = Field(None, description="OCR processing time")
    key_value_extraction_time: Optional[float] = Field(None, description="Key-value extraction processing time")
    status: str = Field(..., description="Overall status of the extraction")
