"""
Pydantic models for face extraction pipeline.
"""
import base64
from pydantic import BaseModel, Field, field_serializer
from typing import List, Optional, Dict, Any
from datetime import datetime


class FaceDetection(BaseModel):
    """Model for detected face information."""
    x: float = Field(..., description="X coordinate of face bounding box")
    y: float = Field(..., description="Y coordinate of face bounding box")
    width: float = Field(..., description="Width of face bounding box")
    height: float = Field(..., description="Height of face bounding box")
    confidence: float = Field(..., description="Confidence score of face detection")
    landmarks: Optional[List[dict]] = Field(None, description="Facial landmarks if available")


class ExtractedFace(BaseModel):
    """Model for extracted face image."""
    face_id: str = Field(..., description="Unique identifier for the extracted face")
    image_data: bytes = Field(..., description="Face image as bytes")
    bounding_box: FaceDetection = Field(..., description="Bounding box information")
    source_document: str = Field(..., description="Source document identifier")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")
    
    @field_serializer('image_data')
    def serialize_image_data(self, value: bytes, _info) -> str:
        """Serialize bytes to base64 string for JSON response."""
        return base64.b64encode(value).decode('utf-8')


class FaceExtractionRequest(BaseModel):
    """Model for face extraction request."""
    document_id: str = Field(..., description="Document identifier")
    extract_all_faces: bool = Field(default=True, description="Extract all faces or just the first")
    min_confidence: float = Field(default=0.7, description="Minimum confidence threshold")


class FaceExtractionResponse(BaseModel):
    """Model for face extraction response."""
    document_id: str = Field(..., description="Document identifier")
    faces_detected: int = Field(..., description="Number of faces detected")
    faces_extracted: List[ExtractedFace] = Field(..., description="List of extracted faces")
    processing_time: float = Field(..., description="Processing time in seconds")
    status: str = Field(..., description="Status of the extraction")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata (e.g., evaluation results)")

