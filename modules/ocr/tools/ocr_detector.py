"""
Tool for performing OCR on images using Google Vision API.
"""
import os
from pathlib import Path
from typing import List, Optional
from google.cloud import vision
from google.oauth2 import service_account
from modules.ocr.models.ocr import TextBlock


class OCRDetector:
    """Tool for detecting and extracting text from images."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OCR detector with Google Vision API.
        
        Args:
            api_key: Optional API key for Vision API (alternative to service account)
        """
        self._vision_client = None
        self.api_key = api_key
    
    def _load_credentials_from_env(self):
        """Load credentials from environment variables."""
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            if not os.path.isabs(creds_path):
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
                credentials = self._load_credentials_from_env()
                
                if credentials:
                    self._vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                else:
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
    
    async def extract_text(
        self,
        image_content: bytes,
        language_hints: Optional[List[str]] = None
    ) -> tuple[str, List[TextBlock], List[str]]:
        """
        Extract text from an image using Google Vision API.
        
        Args:
            image_content: Binary content of the image
            language_hints: Optional list of language codes to help detection
            
        Returns:
            Tuple of (full_text, text_blocks, detected_languages)
        """
        image = vision.Image(content=image_content)
        
        # Configure image context with language hints if provided
        image_context = None
        if language_hints:
            image_context = vision.ImageContext(language_hints=language_hints)
        
        # Use TEXT_DETECTION for dense text (documents)
        features = [vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)]
        
        request = vision.AnnotateImageRequest(
            image=image,
            features=features,
            image_context=image_context
        )
        
        try:
            response = self.vision_client.annotate_image(request=request)
        except Exception as e:
            raise RuntimeError(f"Vision API call failed: {str(e)}") from e
        
        if hasattr(response, 'error') and response.error and hasattr(response.error, 'message') and response.error.message:
            raise RuntimeError(f"Vision API error: {response.error.message}")
        
        # Extract full text
        full_text = ""
        text_blocks = []
        detected_languages = []
        
        if response.text_annotations:
            # First annotation contains the full text
            full_text = response.text_annotations[0].description
            
            # Remaining annotations are individual text blocks with bounding boxes
            for annotation in response.text_annotations[1:]:
                if not hasattr(annotation, 'bounding_poly') or not annotation.bounding_poly:
                    continue
                
                vertices = annotation.bounding_poly.vertices
                if not vertices or len(vertices) < 2:
                    continue
                
                x = min(v.x for v in vertices)
                y = min(v.y for v in vertices)
                width = max(v.x for v in vertices) - x
                height = max(v.y for v in vertices) - y
                
                text_block = TextBlock(
                    text=annotation.description,
                    x=float(x),
                    y=float(y),
                    width=float(width),
                    height=float(height),
                    confidence=None,  # Vision API doesn't provide confidence for text
                    language=None
                )
                text_blocks.append(text_block)
        
        # Try to detect languages (if available in response)
        if hasattr(response, 'text_annotations') and response.text_annotations:
            # Language detection would require additional API call
            # For now, we'll leave it empty or use hints
            if language_hints:
                detected_languages = language_hints
        
        return full_text, text_blocks, detected_languages
    
    async def process_document_for_text(
        self,
        image_content: bytes,
        document_id: str,
        language_hints: Optional[List[str]] = None
    ) -> tuple[str, List[TextBlock], List[str]]:
        """
        Process document and extract all text.
        
        Args:
            image_content: Binary content of the image
            document_id: Document identifier
            language_hints: Optional language hints
            
        Returns:
            Tuple of (full_text, text_blocks, detected_languages)
        """
        return await self.extract_text(image_content, language_hints)

