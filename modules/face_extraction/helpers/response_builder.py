"""
Helper utilities for building face extraction responses from agent/tool context.
"""
import base64
import time
from typing import Dict
from modules.face_extraction.models.face_extraction import FaceExtractionResponse, ExtractedFace


def build_face_response_from_context(ctx: Dict, start_time: float) -> FaceExtractionResponse:
    """Build FaceExtractionResponse from context."""
    extracted_faces_raw = ctx.get('extracted_faces', [])
    
    # Convert to ExtractedFace models, ensuring image_data is bytes
    extracted_faces = []
    for face_dict in extracted_faces_raw:
        # Ensure image_data is bytes, not base64 string
        if isinstance(face_dict.get("image_data"), str):
            face_dict["image_data"] = base64.b64decode(face_dict["image_data"])
        elif not isinstance(face_dict.get("image_data"), bytes):
            continue
        extracted_faces.append(ExtractedFace(**face_dict))
    
    return FaceExtractionResponse(
        document_id=ctx.get('document_id', ''),
        faces_detected=len(extracted_faces),
        faces_extracted=extracted_faces,
        processing_time=time.time() - start_time,
        status="success"
    )

