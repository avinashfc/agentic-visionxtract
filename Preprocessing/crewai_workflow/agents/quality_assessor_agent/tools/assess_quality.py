"""
Quality Assessment Tool for Image Preprocessing.
Uses LLM (Gemini) to analyze image quality and determine preprocessing requirements.
"""

import os
import json
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from crewai.tools import tool

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Configure Gemini
if API_KEY:
    genai.configure(api_key=API_KEY)

# Model name
MODEL_NAME = "gemini-2.5-flash"

QUALITY_ASSESSMENT_PROMPT = """
You are an expert image quality analyst for document OCR preprocessing.

Analyze this document image and assess if it needs preprocessing for OCR.

Evaluate these parameters (0-100 scale):

1. **Brightness** (0-100)
   - GOOD: 40-65 (no preprocessing needed)
   - NEEDS WORK: < 35 (too dark) or > 70 (too bright)
   - adjustment_needed = 0 if within 40-65

2. **Contrast** (0-100)
   - GOOD: 45-75 (no preprocessing needed)
   - NEEDS WORK: < 40 (low contrast, text blends with background)
   - adjustment_needed = 0 if >= 45

3. **Sharpness** (0-100)
   - GOOD: 55+ (text is clear and readable)
   - NEEDS WORK: < 50 (blurry, text edges unclear)
   - adjustment_needed = 0 if >= 55

4. **Noise Level** (0-100)
   - GOOD: < 35 (clean image)
   - NEEDS WORK: > 40 (visible grain/speckles affecting text)
   - adjustment_needed = 0 if < 35

5. **Skew Angle** (degrees)
   - GOOD: -3 to +3 degrees (minor tilt acceptable)
   - NEEDS WORK: > 5 or < -5 degrees (noticeably tilted)
   - adjustment_needed = 0 if within -3 to +3

6. **Resolution** (0-100)
   - GOOD: 55+ (text is clear, not pixelated)
   - NEEDS WORK: < 50 (pixelated or low detail)
   - adjustment_needed = 0 if >= 55

DECISION RULE for preprocessing_required:
- Set to FALSE if ALL parameters are in GOOD range
- Set to TRUE if ANY parameter is in NEEDS WORK range:
  * Brightness < 35 or > 70
  * Contrast < 40
  * Sharpness < 50
  * Noise > 40
  * Skew > 5 or < -5 degrees
  * Resolution < 50

Return ONLY valid JSON (no ```json wrapper):
{
  "quality_scores": {
    "brightness": {"value": <0-100>, "adjustment_needed": <amount or 0>, "confidence_score": <0.0-1.0>},
    "contrast": {"value": <0-100>, "adjustment_needed": <amount or 0>, "confidence_score": <0.0-1.0>},
    "sharpness": {"value": <0-100>, "adjustment_needed": <amount or 0>, "confidence_score": <0.0-1.0>},
    "noise_level": {"value": <0-100>, "adjustment_needed": <amount or 0>, "confidence_score": <0.0-1.0>},
    "skew_angle": {"value": <degrees>, "adjustment_needed": <rotation or 0>, "confidence_score": <0.0-1.0>},
    "resolution": {"value": <0-100>, "adjustment_needed": <amount or 0>, "confidence_score": <0.0-1.0>}
  },
  "preprocessing_required": <true or false>,
  "issues": ["list issues if any"],
  "overall_quality_score": <0-100>,
  "recommendation": "<brief recommendation>"
}
"""


@tool("assess_image_quality")
def assess_image_quality(image_path: str) -> str:
    """
    Assess the quality of an image for OCR readiness using LLM vision analysis.
    
    Analyzes: brightness, contrast, sharpness, noise level, skew angle, and resolution.
    Returns quality scores with adjustment recommendations and confidence scores.
    
    Args:
        image_path: Path to the image file to assess
    
    Returns:
        str: JSON string with quality scores, adjustments needed, and preprocessing recommendations
    """
    image = None
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            return json.dumps({
                "error": f"Image file not found: {image_path}",
                "preprocessing_required": False
            })
        
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Initialize model with generation config
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
        )
        model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)
        
        # Generate response
        response = model.generate_content([QUALITY_ASSESSMENT_PROMPT, image])
        
        # Clean up response
        raw = response.text.strip() if response.text else "{}"
        clean = raw.replace("```json", "").replace("```", "").strip()
        
        try:
            # Validate JSON
            parsed = json.loads(clean)
            
            # Ensure required fields exist
            if "quality_scores" not in parsed:
                parsed["quality_scores"] = {}
            if "preprocessing_required" not in parsed:
                parsed["preprocessing_required"] = False
            if "issues" not in parsed:
                parsed["issues"] = []
            if "overall_quality_score" not in parsed:
                parsed["overall_quality_score"] = 50
            
            # Build recommended_adjustments from quality_scores for compatibility with preprocessor
            recommended_adjustments = {}
            quality_scores = parsed.get("quality_scores", {})
            
            # Only apply adjustments if preprocessing is required
            if parsed.get("preprocessing_required", False):
                # Brightness: if < 35 or > 70
                brightness_adj = quality_scores.get("brightness", {}).get("adjustment_needed", 0)
                if abs(brightness_adj) >= 10:
                    recommended_adjustments["brightness"] = brightness_adj
                
                # Contrast: if < 40
                contrast_adj = quality_scores.get("contrast", {}).get("adjustment_needed", 0)
                if contrast_adj >= 10:
                    recommended_adjustments["contrast"] = 1.0 + (contrast_adj / 50)
                
                # Sharpness: if < 50
                sharp_adj = quality_scores.get("sharpness", {}).get("adjustment_needed", 0)
                if sharp_adj >= 15:
                    recommended_adjustments["sharpening"] = min(2.5, 0.5 + (sharp_adj / 40))
                
                # Noise: if > 40
                noise_adj = quality_scores.get("noise_level", {}).get("adjustment_needed", 0)
                if noise_adj >= 10:
                    recommended_adjustments["denoising"] = min(15, max(3, int(noise_adj / 5)))
                
                # Skew: if > 5 or < -5 degrees
                skew_adj = quality_scores.get("skew_angle", {}).get("adjustment_needed", 0)
                if abs(skew_adj) >= 3:
                    recommended_adjustments["deskew"] = skew_adj
                
                # Resolution: if < 50
                res_adj = quality_scores.get("resolution", {}).get("adjustment_needed", 0)
                if res_adj >= 15:
                    recommended_adjustments["upscale"] = min(2.0, 1.0 + (res_adj / 100))
            
            parsed["recommended_adjustments"] = recommended_adjustments
            
            return json.dumps(parsed)
            
        except json.JSONDecodeError:
            return json.dumps({
                "error": "Failed to parse LLM response",
                "raw_response": clean,
                "preprocessing_required": False
            })
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "preprocessing_required": False
        })
    finally:
        # Explicitly close image to free memory
        if image is not None:
            try:
                image.close()
            except Exception:
                pass
