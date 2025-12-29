import os
import sys
from pathlib import Path
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from crewai.tools import tool
import json

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from preprocessing import enhance_for_stamp_detection

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Configure Gemini
if API_KEY:
    genai.configure(api_key=API_KEY)
    
# Model name
MODEL_NAME = "gemini-2.5-flash"

STAMP_DETECTION_PROMPT = """
Role: You are a forensic document examiner specializing in official stamps, seals, and authentication marks.

Task: Analyze the provided document image with extreme diligence to detect any stamp-like impression, including:
- Colored ink stamps (red, blue, purple, etc.)
- Faint, blurred, smudged, or partial ink impressions
- Text-only marks (e.g., "APPROVED", "RECEIVED") that appear stamped
- Dry/embossed seals (no ink — visible only via texture, shadow, or paper deformation)
- Low-contrast or colorless impressions near signatures, addresses, or official blocks

CRITICAL EXCLUSION:
- IGNORE and DO NOT REPORT any stamps that belong to "Axis Bank" or contain "Axis Bank" text.
- Axis Bank stamps, logos, or seals are part of the document template and should NOT be counted as valid stamps.
- Only detect and report stamps from OTHER companies/entities (e.g., customer company stamps, third-party authorization stamps).
- If the ONLY stamp found is an Axis Bank stamp, respond with "No" for Stamp_Present.

Process:
1. First Pass: Scan the entire document for visible ink-based stamps (excluding Axis Bank stamps).
2. Second Pass (Mandatory): Re-examine corners, signature zones, company name blocks, and date fields for textural anomalies, subtle shadows, or paper indentations — even if no color is present (excluding Axis Bank stamps).
3. Never conclude "no stamp" without completing both passes.

For each detected impression (ink or dry) that is NOT an Axis Bank stamp:
- Bounding Box: (x_min, y_min, x_max, y_max) — as precise as possible
- Type: Ink stamp / Text-only stamp / Dry (embossed) seal
- Text: Extract any legible characters (mark as "[partial]" or "[illegible]" if needed)
- Color & Clarity: e.g., "faded red", "no color – textural only", "blurry blue"
- Location Context: e.g., "below company address", "next to signature"

If absolutely nothing is found after both passes (excluding Axis Bank stamps), respond with "No" for Stamp_Present.
Never ignore texture-based seals (except Axis Bank seals).

CONFIDENCE SCORES:
For each field, provide a confidence_score between 0.0 and 1.0:
- 1.0 = Absolutely certain (stamp clearly visible and identifiable)
- 0.8-0.9 = High confidence (stamp visible but minor ambiguity)
- 0.5-0.7 = Medium confidence (faint impression, partially visible)
- 0.1-0.4 = Low confidence (very faint, possible texture-based detection)
- 0.0 = No stamp found / field is null

Return ONLY a valid JSON object in this EXACT nested format:
{
  "Stamp_Present": {
    "value": "Yes" or "No",
    "confidence_score": <0.0 to 1.0>
  },
  "Stamp_Coordinates": {
    "value": "(x_min, y_min, x_max, y_max)" or null,
    "confidence_score": <0.0 to 1.0>
  },
  "Stamp_Description": {
    "value": "<ONLY the company/entity name>" or null,
    "confidence_score": <0.0 to 1.0>
  }
}

IMPORTANT for Stamp_Description:
- Return ONLY the company name or entity name visible in the stamp
- Do NOT include any additional details like stamp type, color, location, or other metadata
- Just the name/text, nothing else
- Examples: "ABC Corporation Pvt Ltd", "XYZ Industries", "Sharma Enterprises"
- NOT: "Blue ink stamp showing ABC Corporation Pvt Ltd located below signature"

Example outputs:
{"Stamp_Present": {"value": "Yes", "confidence_score": 0.95}, "Stamp_Coordinates": {"value": "(120, 450, 280, 520)", "confidence_score": 0.85}, "Stamp_Description": {"value": "ABC Corporation Pvt Ltd", "confidence_score": 0.75}}
or
{"Stamp_Present": {"value": "No", "confidence_score": 0.9}, "Stamp_Coordinates": {"value": null, "confidence_score": 0.0}, "Stamp_Description": {"value": null, "confidence_score": 0.0}}
"""


def create_field(value, confidence_score: float = 0.0) -> dict:
    """Create a field object with value and confidence_score."""
    return {"value": value, "confidence_score": confidence_score}


def create_error_response(error_msg: str) -> dict:
    """Create an error response with null/No values and zero confidence."""
    return {
        "Stamp_Present": create_field("No", 0.0),
        "Stamp_Coordinates": create_field(None, 0.0),
        "Stamp_Description": create_field(None, 0.0),
        "error": error_msg
    }


@tool("detect_stamp")
def detect_stamp(image_path: str) -> str:
    """
    Detect if a rubber stamp is present on a signature card image using forensic document analysis,
    with confidence scores for each field.
    
    Args:
        image_path: The path to the signature card image file
    
    Returns:
        str: JSON string with nested fields containing value and confidence_score
    """
    image = None
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            return json.dumps(create_error_response(f"Image file not found: {image_path}"))
        
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing optimized for stamp detection (COMMENTED OUT FOR NOW)
        # Enhances faint stamps and embossed seals
        # image = enhance_for_stamp_detection(image)
        
        # Initialize model with generation config
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
        )
        model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)
        
        # Generate response
        response = model.generate_content([STAMP_DETECTION_PROMPT, image])
        
        # Clean up response
        raw = response.text.strip() if response.text else "{}"
        clean = raw.replace("```json", "").replace("```", "").strip()
        
        try:
            # Validate JSON
            parsed = json.loads(clean)
            
            # Ensure all fields have the nested format with value and confidence_score
            for field in ["Stamp_Present", "Stamp_Coordinates", "Stamp_Description"]:
                if field in parsed:
                    # If field is not in nested format, convert it
                    if not isinstance(parsed[field], dict):
                        val = parsed[field]
                        # Normalize Stamp_Present to Yes/No
                        if field == "Stamp_Present":
                            if isinstance(val, bool):
                                val = "Yes" if val else "No"
                            elif isinstance(val, str):
                                val = "Yes" if val.lower() in ["yes", "true"] else "No"
                        parsed[field] = create_field(val, 0.5)
                    else:
                        # Ensure both keys exist
                        if "value" not in parsed[field]:
                            parsed[field]["value"] = None if field != "Stamp_Present" else "No"
                        if "confidence_score" not in parsed[field]:
                            parsed[field]["confidence_score"] = 0.0 if parsed[field]["value"] is None else 0.5
                        # Normalize Stamp_Present value
                        if field == "Stamp_Present":
                            val = parsed[field]["value"]
                            if isinstance(val, bool):
                                parsed[field]["value"] = "Yes" if val else "No"
                            elif isinstance(val, str):
                                parsed[field]["value"] = "Yes" if val.lower() in ["yes", "true"] else "No"
                else:
                    default_val = "No" if field == "Stamp_Present" else None
                    parsed[field] = create_field(default_val, 0.0)
            
            return json.dumps(parsed)
        except json.JSONDecodeError:
            # Try to infer from raw response
            lower_response = clean.lower()
            if "yes" in lower_response or "true" in lower_response or "present" in lower_response:
                response = {
                    "Stamp_Present": create_field("Yes", 0.5),
                    "Stamp_Coordinates": create_field(None, 0.0),
                    "Stamp_Description": create_field(None, 0.0),
                    "raw_response": clean
                }
                return json.dumps(response)
            elif "no" in lower_response or "false" in lower_response or "not present" in lower_response:
                return json.dumps({
                    "Stamp_Present": create_field("No", 0.5),
                    "Stamp_Coordinates": create_field(None, 0.0),
                    "Stamp_Description": create_field(None, 0.0)
                })
            else:
                response = create_error_response("Failed to parse LLM response")
                response["raw_response"] = clean
                return json.dumps(response)
        
    except Exception as e:
        return json.dumps(create_error_response(str(e)))
    finally:
        # Explicitly close image to free memory
        if image is not None:
            try:
                image.close()
            except Exception:
                pass

