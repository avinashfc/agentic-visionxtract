from crewai import Crew, Task, Process
from crewai_workflow.agents.extractor_agent.agent import ExtractorAgent
from crewai_workflow.agents.stamp_detector_agent.agent import StampDetectorAgent
from crewai_workflow.agents.extractor_agent.tools.extract_data import extract_signature_card_data
from crewai_workflow.agents.stamp_detector_agent.tools.detect_stamp import detect_stamp
from PIL import Image
import tempfile
import os
import base64
import gc
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Singleton instance to avoid recreating agents and LLMs on every request
_signature_card_crew_instance = None


class SignatureCardCrew:
    def __init__(self):
        self.extractor = ExtractorAgent()
        self.stamp_detector = StampDetectorAgent()
    
    @classmethod
    def get_instance(cls):
        """Get or create singleton instance to reuse agents and LLMs."""
        global _signature_card_crew_instance
        if _signature_card_crew_instance is None:
            _signature_card_crew_instance = cls()
        return _signature_card_crew_instance

    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _parse_task_output(self, output) -> dict:
        """Parse task output to dictionary, handling various formats."""
        if output is None:
            return {}
        
        # If it has raw attribute (TaskOutput object)
        if hasattr(output, 'raw'):
            output = output.raw
        
        # If already a dict, return it
        if isinstance(output, dict):
            return output
        
        # Convert to string and parse
        output_str = str(output).strip()
        
        # Remove markdown code blocks
        output_str = re.sub(r'^```json\s*', '', output_str)
        output_str = re.sub(r'^```\s*', '', output_str)
        output_str = re.sub(r'\s*```$', '', output_str)
        output_str = output_str.strip()
        
        try:
            return json.loads(output_str)
        except json.JSONDecodeError:
            # Try to find JSON object in the string
            json_match = re.search(r'\{[^{}]*\}', output_str, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {"raw": output_str}

    def process_signature_card(self, image: Image.Image):
        """
        Process a signature card image to extract:
        - Name, PAN, Barcode number (via Extractor Agent)
        - Stamp presence (via Stamp Detector Agent)
        
        Both tasks run in PARALLEL for faster processing.
        """
        try:
            # Save image to temporary file for multimodal input
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                # Convert to RGB if necessary (for PNG with transparency)
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                image.save(tmp_file.name, format="PNG")
                image_path = tmp_file.name
            
            try:
                # Get absolute path for the image
                abs_image_path = str(Path(image_path).absolute())
                
                # Run both tools in parallel using ThreadPoolExecutor
                final_result = {}
                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit both tasks to run in parallel
                    # Use .run() method since @tool decorator creates Tool objects
                    extract_future = executor.submit(extract_signature_card_data.run, abs_image_path)
                    stamp_future = executor.submit(detect_stamp.run, abs_image_path)
                    
                    # Collect results as they complete
                    for future in as_completed([extract_future, stamp_future]):
                        try:
                            result = future.result()
                            parsed = self._parse_task_output(result)
                            if isinstance(parsed, dict):
                                final_result.update(parsed)
                        except Exception as e:
                            print(f"Task error: {e}")
                
                gc.collect()
                return final_result
                
            finally:
                # Clean up temporary file
                if 'image_path' in locals() and os.path.exists(image_path):
                    try:
                        os.unlink(image_path)
                    except Exception as e:
                        print(f"Warning: Could not delete temp file {image_path}: {e}")
                
                gc.collect()
            
        except Exception as e:
            print(f"Error processing signature card: {e}")
            import traceback
            traceback.print_exc()
            return None

