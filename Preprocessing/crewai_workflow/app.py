"""
Preprocessing Crew - Image Quality Assessment and Enhancement Pipeline.

Flow:
1. Quality Assessment Agent analyzes image quality
2. If preprocessing needed → Preprocessor Agent enhances image
3. Re-assess quality after preprocessing
4. If still needs preprocessing → Return error (image too poor quality)
5. If quality OK → Proceed to OCR
"""

from crewai_workflow.agents.quality_assessor_agent.agent import QualityAssessorAgent
from crewai_workflow.agents.preprocessor_agent.agent import PreprocessorAgent
from crewai_workflow.agents.quality_assessor_agent.tools.assess_quality import assess_image_quality
from crewai_workflow.agents.preprocessor_agent.tools.preprocess_image import preprocess_image
from PIL import Image
import tempfile
import os
import json
import gc
from pathlib import Path

# Singleton instance
_preprocessing_crew_instance = None


class PreprocessingCrew:
    def __init__(self):
        self.quality_assessor = QualityAssessorAgent()
        self.preprocessor = PreprocessorAgent()
    
    @classmethod
    def get_instance(cls):
        """Get or create singleton instance to reuse agents."""
        global _preprocessing_crew_instance
        if _preprocessing_crew_instance is None:
            _preprocessing_crew_instance = cls()
        return _preprocessing_crew_instance

    def _parse_json_result(self, result) -> dict:
        """Parse JSON string or dict result."""
        if isinstance(result, dict):
            return result
        try:
            return json.loads(str(result))
        except json.JSONDecodeError:
            return {"error": "Failed to parse result", "raw": str(result)}

    def process_image(self, image: Image.Image) -> dict:
        """
        Process an image through the preprocessing pipeline.
        
        Flow:
        1. Assess quality
        2. If preprocessing needed → Apply preprocessing → Re-assess
        3. If still needs preprocessing → Return error
        4. If quality OK → Return success (ready for OCR)
        
        Args:
            image: PIL Image to process
            
        Returns:
            dict with:
                - status: "ready_for_ocr" | "preprocessing_applied" | "image_rejected"
                - image_path: Path to final image (original or preprocessed)
                - quality_scores: Final quality assessment
                - message: Status message
        """
        temp_files = []  # Track temp files for cleanup
        
        try:
            # Save input image to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                image.save(tmp_file.name, format="PNG")
                original_image_path = tmp_file.name
                temp_files.append(original_image_path)
            
            current_image_path = original_image_path
            
            print("\n" + "=" * 60)
            print("PREPROCESSING PIPELINE - PASS 1")
            print("=" * 60)
            
            # Step 1: Initial Quality Assessment
            print("\n[Step 1] Assessing image quality...")
            assessment_result = assess_image_quality.run(current_image_path)
            assessment = self._parse_json_result(assessment_result)
            
            if "error" in assessment:
                return {
                    "status": "error",
                    "message": f"Quality assessment failed: {assessment['error']}",
                    "image_path": original_image_path
                }
            
            print(f"  Quality Scores: {assessment.get('quality_scores', {})}")
            print(f"  Issues Found: {assessment.get('issues', [])}")
            print(f"  Preprocessing Required: {assessment.get('preprocessing_required', False)}")
            
            # Step 2: Check if preprocessing is needed
            if not assessment.get("preprocessing_required", False):
                print("\n✅ Image quality is good. No preprocessing required.")
                return {
                    "status": "ready_for_ocr",
                    "message": "No preprocessing required. Image quality is sufficient for OCR.",
                    "image_path": original_image_path,
                    "quality_scores": assessment.get("quality_scores", {}),
                    "preprocessing_applied": False
                }
            
            # Step 3: Apply Preprocessing
            print("\n[Step 2] Applying preprocessing...")
            adjustments = assessment.get("recommended_adjustments", {})
            adjustments_json = json.dumps(adjustments)
            
            print(f"  Adjustments to apply: {adjustments}")
            
            preprocess_result = preprocess_image.run(current_image_path, adjustments_json)
            preprocess_data = self._parse_json_result(preprocess_result)
            
            if not preprocess_data.get("success", False):
                return {
                    "status": "error",
                    "message": f"Preprocessing failed: {preprocess_data.get('error', 'Unknown error')}",
                    "image_path": original_image_path
                }
            
            preprocessed_image_path = preprocess_data.get("preprocessed_image_path")
            temp_files.append(preprocessed_image_path)
            current_image_path = preprocessed_image_path
            
            print(f"  Applied: {preprocess_data.get('applied_adjustments', [])}")
            print(f"  Preprocessed image saved to: {preprocessed_image_path}")
            
            # Step 4: Re-assess quality after preprocessing
            print("\n" + "=" * 60)
            print("PREPROCESSING PIPELINE - PASS 2 (Re-assessment)")
            print("=" * 60)
            
            print("\n[Step 3] Re-assessing image quality after preprocessing...")
            reassessment_result = assess_image_quality.run(current_image_path)
            reassessment = self._parse_json_result(reassessment_result)
            
            if "error" in reassessment:
                return {
                    "status": "error",
                    "message": f"Re-assessment failed: {reassessment['error']}",
                    "image_path": preprocessed_image_path
                }
            
            print(f"  Quality Scores: {reassessment.get('quality_scores', {})}")
            print(f"  Issues Remaining: {reassessment.get('issues', [])}")
            print(f"  Still Needs Preprocessing: {reassessment.get('preprocessing_required', False)}")
            
            # Step 5: Final Decision
            if reassessment.get("preprocessing_required", False):
                # Image still not good enough after preprocessing
                print("\n❌ Image quality is still insufficient after preprocessing.")
                print("   Please upload a clearer image.")
                return {
                    "status": "image_rejected",
                    "message": "Image quality is still insufficient after preprocessing. Please upload a clearer, higher quality image.",
                    "image_path": None,
                    "quality_scores": reassessment.get("quality_scores", {}),
                    "remaining_issues": reassessment.get("issues", []),
                    "preprocessing_applied": True,
                    "adjustments_applied": preprocess_data.get("applied_adjustments", [])
                }
            else:
                # Image is now good for OCR
                print("\n✅ Preprocessing successful! Image is now ready for OCR.")
                return {
                    "status": "ready_for_ocr",
                    "message": "Preprocessing applied successfully. Image is now ready for OCR.",
                    "image_path": preprocessed_image_path,
                    "quality_scores": reassessment.get("quality_scores", {}),
                    "preprocessing_applied": True,
                    "adjustments_applied": preprocess_data.get("applied_adjustments", [])
                }
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Pipeline error: {str(e)}",
                "image_path": None
            }
        finally:
            gc.collect()
    
    def cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files."""
        for path in file_paths:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {path}: {e}")


# Convenience function for direct usage
def process_image_for_ocr(image: Image.Image) -> dict:
    """
    Process an image through the preprocessing pipeline.
    
    Args:
        image: PIL Image to process
        
    Returns:
        dict with status, message, image_path, and quality information
    """
    crew = PreprocessingCrew.get_instance()
    return crew.process_image(image)

