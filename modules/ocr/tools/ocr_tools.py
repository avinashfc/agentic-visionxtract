"""
ADK Tools for OCR.
These tools wrap the underlying functionality for use with Google ADK agents.
"""
import json
import io
import time
import logging
from typing import List, Optional, Any
from PIL import Image
from google.adk.tools import FunctionTool
from google.genai import Client, types
from modules.ocr.tools.ocr_detector import OCRDetector
from modules.ocr.models.ocr import TextBlock

logger = logging.getLogger(__name__)


class OCRTools:
    """Collection of ADK tools for OCR."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize OCR tools.
        
        Args:
            api_key: Optional API key for Vision API and Gemini
            model_name: Gemini model name for key-value extraction (must be provided)
        """
        if not model_name:
            raise ValueError("model_name must be provided. Configure it in config.yaml or pass it explicitly.")
        self.ocr_detector = OCRDetector(api_key=api_key)
        self.api_key = api_key
        self.model_name = model_name
        # Context for agentic execution
        self._context: dict[str, Any] = {}
        
        # Initialize genai client for key-value extraction
        self.genai_client = None
        if api_key:
            try:
                self.genai_client = Client(api_key=api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")
                logger.warning("Key-value extraction will not be available without Gemini client.")
    
    async def validate_document(self, document_name: str) -> dict:
        """
        Validate an uploaded document/image.
        The file content is accessed from the execution context.
        
        Args:
            document_name: Name of the document
            
        Returns:
            Dictionary with validation results
        """
        file_content = self._context.get('file_content')
        if not file_content:
            return {
                "valid": False,
                "document_name": document_name,
                "error": "File content not available in context"
            }
        
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
        Generate document ID for tracking.
        The file content is accessed from the execution context.
        
        Args:
            document_name: Name of the document
            
        Returns:
            Dictionary with document information including document_id
        """
        file_content = self._context.get('file_content')
        if not file_content:
            return {
                "document_id": "",
                "error": "File content not available in context"
            }
        
        document_id = f"doc_{document_name}_{int(time.time() * 1000) % 1000000}"
        
        self._context['document_id'] = document_id
        
        return {
            "document_id": document_id,
            "document_name": document_name,
            "file_size": len(file_content)
        }
    
    async def extract_text(
        self,
        language_hints_json: Optional[str] = None
    ) -> dict:
        """
        Extract text from an image using OCR.
        The image content is accessed from the execution context.
        
        Args:
            language_hints_json: Optional JSON string of language codes (e.g., '["en", "es"]')
            
        Returns:
            Dictionary with extracted text and text blocks
        """
        image_content = self._context.get('file_content')
        if not image_content:
            return {
                "full_text": "",
                "text_blocks": [],
                "error": "File content not available in context"
            }
        
        language_hints = None
        if language_hints_json:
            try:
                language_hints = json.loads(language_hints_json)
            except json.JSONDecodeError:
                pass
        
        full_text, text_blocks, detected_languages = await self.ocr_detector.extract_text(
            image_content,
            language_hints=language_hints
        )
        
        # Store results in context
        self._context['full_text'] = full_text
        self._context['text_blocks'] = [block.model_dump() for block in text_blocks]
        self._context['detected_languages'] = detected_languages
        
        return {
            "full_text": full_text,
            "text_blocks": [block.model_dump() for block in text_blocks],
            "detected_languages": detected_languages,
            "text_block_count": len(text_blocks)
        }
    
    async def extract_key_value_pairs(
        self,
        extraction_prompt: Optional[str] = None
    ) -> dict:
        """
        Extract key-value pairs from OCR text using Gemini LLM.
        The OCR text is accessed from the execution context.
        
        Args:
            extraction_prompt: Optional custom prompt for extraction. If not provided,
                            a default prompt will be used to extract all key-value pairs.
                            
        Returns:
            Dictionary with extracted key-value pairs
        """
        full_text = self._context.get('full_text')
        if not full_text:
            return {
                "key_value_pairs": [],
                "error": "OCR text not available in context. Please run extract_text first."
            }
        
        if not self.genai_client:
            return {
                "key_value_pairs": [],
                "error": "Gemini client not initialized. API key required for key-value extraction."
            }
        
        # Default prompt if none provided
        if not extraction_prompt:
            extraction_prompt = """Analyze the following OCR text and extract all key-value pairs.
A key-value pair consists of a label/key (like "Name", "Date", "Amount", etc.) and its corresponding value.

Extract all meaningful key-value pairs from the text. Keys should be descriptive labels (e.g., "Invoice Number", "Total Amount", "Customer Name").
Values should be the actual data corresponding to each key.

Return the results as a JSON array of objects, where each object has:
- "key": the label/field name
- "value": the corresponding value
- "confidence": optional confidence score (0.0 to 1.0)

Example format:
[
  {{"key": "Invoice Number", "value": "INV-2024-001", "confidence": 0.95}},
  {{"key": "Date", "value": "2024-01-15", "confidence": 0.90}},
  {{"key": "Total Amount", "value": "$1,250.00", "confidence": 0.85}}
]

OCR Text:
{ocr_text}

Extract all key-value pairs and return only valid JSON, no additional text."""
        
        # Format the prompt with OCR text
        formatted_prompt = extraction_prompt.format(ocr_text=full_text)
        
        try:
            # Call Gemini to extract key-value pairs
            # Use types.Content and types.Part for proper message format
            content = types.Content(parts=[types.Part(text=formatted_prompt)])
            
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=[content]
            )
            
            # Extract text from response
            if (response.candidates and 
                len(response.candidates) > 0 and 
                response.candidates[0].content and
                response.candidates[0].content.parts and
                len(response.candidates[0].content.parts) > 0):
                response_text = response.candidates[0].content.parts[0].text.strip()
            else:
                return {
                    "key_value_pairs": [],
                    "error": "No content in Gemini response",
                    "raw_response": str(response)
                }
            
            # Try to extract JSON from the response
            # Sometimes LLM adds markdown code blocks or extra text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON response
            try:
                key_value_pairs = json.loads(response_text)
                if not isinstance(key_value_pairs, list):
                    key_value_pairs = [key_value_pairs] if key_value_pairs else []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Gemini response: {e}")
                logger.debug(f"Response text: {response_text}")
                return {
                    "key_value_pairs": [],
                    "error": f"Failed to parse JSON response: {str(e)}",
                    "raw_response": response_text
                }
            
            # Store results in context
            self._context['key_value_pairs'] = key_value_pairs
            
            return {
                "key_value_pairs": key_value_pairs,
                "count": len(key_value_pairs),
                "raw_text": full_text
            }
            
        except Exception as e:
            logger.error(f"Error extracting key-value pairs: {e}", exc_info=True)
            return {
                "key_value_pairs": [],
                "error": f"Error during key-value extraction: {str(e)}"
            }
    
    def get_tools(self) -> List[FunctionTool]:
        """
        Get list of ADK FunctionTool instances.
        
        Returns:
            List of FunctionTool instances
        """
        tools = [
            FunctionTool(self.validate_document),
            FunctionTool(self.upload_document),
            FunctionTool(self.extract_text),
        ]
        
        # Add key-value extraction tool if Gemini client is available
        if self.genai_client:
            tools.append(FunctionTool(self.extract_key_value_pairs))
        
        return tools

