"""
Pydantic models for OCR pipeline.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class TextBlock(BaseModel):
    """Model for detected text block."""
    text: str = Field(..., description="Extracted text content")
    x: float = Field(..., description="X coordinate of text bounding box")
    y: float = Field(..., description="Y coordinate of text bounding box")
    width: float = Field(..., description="Width of text bounding box")
    height: float = Field(..., description="Height of text bounding box")
    confidence: Optional[float] = Field(None, description="Confidence score if available")
    language: Optional[str] = Field(None, description="Detected language if available")


class OCRRequest(BaseModel):
    """Model for OCR request."""
    document_id: str = Field(..., description="Document identifier")
    language_hints: Optional[List[str]] = Field(default=None, description="Optional language hints for better detection")


class OCRResponse(BaseModel):
    """Model for OCR response."""
    document_id: str = Field(..., description="Document identifier")
    full_text: str = Field(..., description="Complete extracted text from document")
    text_blocks: List[TextBlock] = Field(..., description="List of detected text blocks with positions")
    languages_detected: List[str] = Field(default_factory=list, description="Detected languages")
    processing_time: float = Field(..., description="Processing time in seconds")
    status: str = Field(..., description="Status of the OCR extraction")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata (e.g., evaluation results)")


class KeyValuePair(BaseModel):
    """Model for a key-value pair extracted from OCR text."""
    key: str = Field(..., description="The key/label of the extracted field")
    value: str = Field(..., description="The value corresponding to the key")
    confidence: Optional[float] = Field(None, description="Confidence score for the extraction if available")


class KeyValueResponse(BaseModel):
    """Model for key-value extraction response."""
    document_id: str = Field(..., description="Document identifier")
    key_value_pairs: List[KeyValuePair] = Field(..., description="List of extracted key-value pairs")
    raw_text: str = Field(..., description="Original OCR text used for extraction")
    processing_time: float = Field(..., description="Processing time in seconds")
    status: str = Field(..., description="Status of the key-value extraction")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata (e.g., evaluation results)")

