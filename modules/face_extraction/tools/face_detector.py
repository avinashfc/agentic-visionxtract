"""
Tool for detecting and extracting faces from images using Google Vision API.
"""
import os
import base64
import io
from pathlib import Path
from typing import List, Optional
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
from modules.face_extraction.models.face_extraction import FaceDetection, ExtractedFace


class FaceDetector:
    """Tool for detecting and extracting faces from images."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize face detector with Google Vision API.
        
        Args:
            api_key: Optional API key for Vision API (alternative to service account)
        """
        # Lazy initialization to avoid credential errors at import time
        self._vision_client = None
        self.api_key = api_key
    
    def _load_credentials_from_env(self):
        """Load credentials from environment variables."""
        # Check for GOOGLE_APPLICATION_CREDENTIALS
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            # Handle relative paths
            if not os.path.isabs(creds_path):
                # Try relative to project root
                project_root = Path(__file__).parent.parent.parent.parent
                abs_path = project_root / creds_path
                if abs_path.exists():
                    creds_path = str(abs_path)
            
            if os.path.exists(creds_path):
                return service_account.Credentials.from_service_account_file(creds_path)
            else:
                raise FileNotFoundError(
                    f"Credentials file not found: {creds_path}\n"
                    f"Please check GOOGLE_APPLICATION_CREDENTIALS in your .env file."
                )
        return None
    
    @property
    def vision_client(self):
        """Get Vision API client (lazy initialization)."""
        if self._vision_client is None:
            try:
                # Try to load credentials from environment
                credentials = self._load_credentials_from_env()
                
                if credentials:
                    # Use explicit credentials
                    self._vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                else:
                    # Try default credentials (for gcloud auth application-default login)
                    self._vision_client = vision.ImageAnnotatorClient()
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Failed to initialize Google Vision API client: {str(e)}\n"
                    "Please check your .env file and ensure GOOGLE_APPLICATION_CREDENTIALS points to a valid service account JSON file."
                ) from e
            except Exception as e:
                error_msg = (
                    f"Failed to initialize Google Vision API client: {str(e)}\n\n"
                    "Please ensure one of the following is set up:\n"
                    "1. Set GOOGLE_APPLICATION_CREDENTIALS in .env file to path of service account JSON\n"
                    "2. Run: gcloud auth application-default login\n"
                    "3. Set GOOGLE_APPLICATION_CREDENTIALS as environment variable\n\n"
                    "See: https://cloud.google.com/docs/authentication/external/set-up-adc"
                )
                raise RuntimeError(error_msg) from e
        return self._vision_client
    
    async def detect_faces(
        self,
        image_content: bytes,
        min_confidence: float = 0.7,
        max_results: Optional[int] = None
    ) -> List[FaceDetection]:
        """
        Detect faces in an image using Google Vision API.
        
        Args:
            image_content: Binary content of the image
            min_confidence: Minimum confidence threshold
            max_results: Maximum number of faces to detect
            
        Returns:
            List of FaceDetection objects
        """
        # Prepare image for Vision API
        image = vision.Image(content=image_content)
        
        # Configure face detection features
        features = [vision.Feature(type_=vision.Feature.Type.FACE_DETECTION)]
        
        # Create request
        request = vision.AnnotateImageRequest(
            image=image,
            features=features
        )
        
        # Perform face detection
        try:
            response = self.vision_client.annotate_image(request=request)
        except Exception as e:
            raise RuntimeError(f"Vision API call failed: {str(e)}") from e
        
        # Check for errors in response
        if hasattr(response, 'error') and response.error and hasattr(response.error, 'message') and response.error.message:
            raise RuntimeError(f"Vision API error: {response.error.message}")
        
        faces = []
        
        if response.face_annotations:
            for idx, face in enumerate(response.face_annotations[:max_results] if max_results else response.face_annotations):
                # Check confidence threshold
                # detection_confidence can be 0-100 (integer) or already 0-1 (float)
                raw_confidence = face.detection_confidence if face.detection_confidence else 0.0
                
                # Determine if it's 0-100 scale (if > 1, it's 0-100 scale)
                if raw_confidence > 1.0:
                    detection_confidence = raw_confidence / 100.0
                else:
                    detection_confidence = raw_confidence
                
                if detection_confidence < min_confidence:
                    continue
                
                # Get bounding box
                if not hasattr(face, 'bounding_poly') or not face.bounding_poly:
                    continue
                    
                vertices = face.bounding_poly.vertices
                if not vertices or len(vertices) < 2:
                    continue
                
                x = min(v.x for v in vertices)
                y = min(v.y for v in vertices)
                width = max(v.x for v in vertices) - x
                height = max(v.y for v in vertices) - y
                
                # Extract landmarks if available
                landmarks = []
                if hasattr(face, 'landmarks') and face.landmarks:
                    for landmark in face.landmarks:
                        landmark_dict = {
                            "x": landmark.position.x,
                            "y": landmark.position.y,
                            "z": landmark.position.z
                        }
                        # Try to get landmark type
                        if hasattr(landmark, 'type_'):
                            landmark_dict["type"] = str(landmark.type_)
                        landmarks.append(landmark_dict)
                
                face_detection = FaceDetection(
                    x=float(x),
                    y=float(y),
                    width=float(width),
                    height=float(height),
                    confidence=detection_confidence,
                    landmarks=landmarks if landmarks else None
                )
                faces.append(face_detection)
        
        return faces
    
    async def extract_face_images(
        self,
        image_content: bytes,
        face_detections: List[FaceDetection],
        document_id: str
    ) -> List[ExtractedFace]:
        """
        Extract face images from the original image based on detections.
        
        Args:
            image_content: Binary content of the original image
            face_detections: List of face detection results
            document_id: Source document identifier
            
        Returns:
            List of ExtractedFace objects
        """
        # Load original image
        original_image = Image.open(io.BytesIO(image_content))
        
        extracted_faces = []
        for idx, face_detection in enumerate(face_detections):
            # Crop face from image
            x = int(face_detection.x)
            y = int(face_detection.y)
            width = int(face_detection.width)
            height = int(face_detection.height)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(original_image.width - x, width + 2 * padding)
            height = min(original_image.height - y, height + 2 * padding)
            
            face_image = original_image.crop((x, y, x + width, y + height))
            
            # Convert to bytes
            face_buffer = io.BytesIO()
            face_image.save(face_buffer, format='PNG')
            face_image_bytes = face_buffer.getvalue()
            
            # Create extracted face object
            face_id = f"{document_id}_face_{idx + 1}"
            extracted_face = ExtractedFace(
                face_id=face_id,
                image_data=face_image_bytes,
                bounding_box=face_detection,
                source_document=document_id
            )
            extracted_faces.append(extracted_face)
        
        return extracted_faces
    
    async def process_document_for_faces(
        self,
        image_content: bytes,
        document_id: str,
        min_confidence: float = 0.7,
        extract_all: bool = True
    ) -> List[ExtractedFace]:
        """
        Complete pipeline: detect and extract faces from document.
        
        Args:
            image_content: Binary content of the document image
            document_id: Document identifier
            min_confidence: Minimum confidence threshold
            extract_all: Whether to extract all faces or just the first
            
        Returns:
            List of ExtractedFace objects
        """
        # Detect faces
        face_detections = await self.detect_faces(
            image_content,
            min_confidence=min_confidence,
            max_results=None if extract_all else 1
        )
        
        if not face_detections:
            return []
        
        # Extract face images
        extracted_faces = await self.extract_face_images(
            image_content,
            face_detections,
            document_id
        )
        
        return extracted_faces

