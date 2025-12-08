"""
ADK Tools for face extraction.
These tools wrap the underlying functionality for use with Google ADK agents.
"""
import json
import io
import time
from typing import List, Optional, Any
from PIL import Image
from google.adk.tools import FunctionTool
from modules.face_extraction.tools.face_detector import FaceDetector
from modules.face_extraction.models.face_extraction import FaceDetection, ExtractedFace


class FaceExtractionTools:
    """Collection of ADK tools for face extraction."""
    
    def __init__(
        self,
        api_key: Optional[str] = None
    ):
        """
        Initialize face extraction tools.
        
        Args:
            api_key: Optional API key for Vision API
        """
        self.face_detector = FaceDetector(api_key=api_key)
    
    async def validate_document(self, document_name: str) -> dict:
        """
        Validate an uploaded document/image.
        The file content is accessed from the execution context.
        
        Args:
            document_name: Name of the document
            
        Returns:
            Dictionary with validation results
        """
        # Get file content from context (set by agent before execution)
        file_content = getattr(self, '_context', {}).get('file_content')
        if not file_content:
            return {
                "valid": False,
                "document_name": document_name,
                "error": "File content not available in context"
            }
        
        # Validate image using PIL
        try:
            image = Image.open(io.BytesIO(file_content))
            return {
                "valid": True,
                "document_name": document_name,
                "format": image.format,
                "size": image.size,
                "mode": image.mode
            }
        except Exception as e:
            return {
                "valid": False,
                "document_name": document_name,
                "error": str(e)
            }
    
    async def upload_document(self, document_name: str) -> dict:
        """
        Generate document ID for tracking faces.
        The file content is accessed from the execution context.
        
        Note: We don't upload the image - it's already in memory.
        This function just generates a document_id for tracking/identifying extracted faces.
        
        Args:
            document_name: Name of the document
            
        Returns:
            Dictionary with document information including document_id
        """
        # Get file content from context
        file_content = getattr(self, '_context', {}).get('file_content')
        if not file_content:
            return {
                "document_id": "",
                "error": "File content not available in context"
            }
        
        # Generate document_id (needed for tracking/identifying faces)
        document_id = f"doc_{document_name}_{int(time.time() * 1000) % 1000000}"
        
        result = {
            "document_id": document_id,
            "document_name": document_name,
            "file_size": len(file_content)
        }
        
        # Store document_id in context for later use (face extraction needs it)
        if hasattr(self, '_context'):
            self._context['document_id'] = document_id
        
        return result
    
    async def detect_faces(
        self,
        min_confidence: float = 0.7,
        max_results: Optional[int] = None
    ) -> List[dict]:
        """
        Detect faces in an image.
        The image content is accessed from the execution context.
        
        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0)
            max_results: Maximum number of faces to detect (None for all)
            
        Returns:
            List of face detection dictionaries
        """
        # Get file content and parameters from context
        ctx = getattr(self, '_context', {})
        image_content = ctx.get('file_content')
        if not image_content:
            return []
        
        # Use context parameters if not provided
        if min_confidence == 0.7:  # Default value, check context
            min_confidence = ctx.get('min_confidence', min_confidence)
        if max_results is None:
            extract_all = ctx.get('extract_all_faces', True)
            max_results = None if extract_all else 1
        
        faces = await self.face_detector.detect_faces(
            image_content,
            min_confidence=min_confidence,
            max_results=max_results
        )
        
        # Store face detections in context for later use
        if hasattr(self, '_context'):
            self._context['face_detections'] = [face.model_dump() for face in faces]
        
        # Convert Pydantic models to dicts for ADK
        return [face.model_dump() for face in faces]
    
    async def extract_face_images(
        self,
        face_detections_json: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> List[dict]:
        """
        Extract face images from the original image based on detections.
        The image content and detections are accessed from the execution context.
        
        Args:
            face_detections_json: Optional JSON string of face detection dictionaries.
                                 If not provided, uses detections from context.
            document_id: Optional source document identifier. If not provided, uses from context.
            
        Returns:
            List of extracted face dictionaries
        """
        # Get data from context
        ctx = getattr(self, '_context', {})
        image_content = ctx.get('file_content')
        if not image_content:
            return []
        
        # Get document_id from context if not provided
        if not document_id:
            document_id = ctx.get('document_id', '')
        
        # Get face detections from context if not provided
        if not face_detections_json:
            face_detections = ctx.get('face_detections', [])
        else:
            # Parse JSON string to list of dicts
            face_detections = json.loads(face_detections_json)
        
        if not face_detections:
            return []
        
        # Convert dicts back to Pydantic models
        face_models = [FaceDetection(**fd) for fd in face_detections]
        
        extracted_faces = await self.face_detector.extract_face_images(
            image_content,
            face_models,
            document_id
        )
        
        # Store extracted faces in context - keep image_data as bytes for internal use
        if hasattr(self, '_context'):
            # Store with bytes (not base64) so it can be properly converted later
            self._context['extracted_faces'] = [face.model_dump() for face in extracted_faces]
        
        # Convert to dicts and encode image data as base64 for function response
        # (ADK function responses need to be JSON-serializable)
        import base64
        result = []
        for face in extracted_faces:
            face_dict = face.model_dump()
            # Convert bytes to base64 string for JSON serialization in function response
            if isinstance(face_dict.get("image_data"), bytes):
                face_dict["image_data"] = base64.b64encode(face_dict["image_data"]).decode('utf-8')
            result.append(face_dict)
        
        return result
    
    def get_tools(self) -> List[FunctionTool]:
        """
        Get list of ADK FunctionTool instances.
        
        Returns:
            List of FunctionTool instances
        """
        return [
            FunctionTool(self.validate_document),
            FunctionTool(self.upload_document),
            FunctionTool(self.detect_faces),
            FunctionTool(self.extract_face_images),
        ]

