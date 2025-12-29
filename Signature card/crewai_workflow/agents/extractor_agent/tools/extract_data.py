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
from preprocessing import enhance_handwritten_text

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Configure Gemini
if API_KEY:
    genai.configure(api_key=API_KEY)
    
# Model name
MODEL_NAME = "gemini-2.5-flash"

EXTRACTION_PROMPT = """
You are an expert at extracting handwritten data from signature card documents.
Extract ONLY the following data from this signature card:

1. Customer Name → key: "Customer_Name"
2. PAN number → key: "PAN_Number" (Always 10 characters: 5 alphabets, 4 digits, 1 alphabet, e.g., ABCDE1234F)
3. Barcode Number → key: "Barcode_Number" (Located in top right of the document)
4. Barcode Formatting → key: "Barcode_Formatting"

IMPORTANT - Document Format:
- The data is written in CHARACTER BOXES (comb fields) where each character occupies a separate box.
- Text is written in BLOCK LETTERS (capital letters, handwritten).
- Read each character box individually and combine them to form the complete value.

CRITICAL - Distinguishing Handwriting from Box Borders:
- The boxes have THIN, STRAIGHT, UNIFORM printed lines (usually gray or light black) forming a grid.
- Handwritten characters have THICKER, IRREGULAR ink strokes with varying pressure and slight curves.
- Box borders are perfectly vertical/horizontal lines; handwriting has natural variations.
- IGNORE all straight uniform lines that form the box grid - these are NOT characters.
- If a "character" looks exactly like a box border line, it is likely NOT a character - skip it.
- Empty boxes should be read as spaces, not as characters.

CRITICAL - Common Character Misreads to AVOID:
- V vs N: The letter "V" has TWO DIAGONAL strokes meeting at the bottom. Do NOT add a vertical line from the box border to make it look like "N". If you see two diagonal strokes forming a "V" shape, it is V, not N.
- Single characters at END of name: ALWAYS check for single letters (like initials) at the end of the name. Names like "GIREESH KRISHNAN G" have a trailing "G" - do NOT drop it.
- Repeated single letters with spaces: Names like "MUNEER V V" or "SAJID RAHMAN V A" have single letters separated by spaces. Preserve the spaces between them. Do NOT merge "V V" into "VV".
- N vs NC: The letter "N" is a single character with two vertical strokes and one diagonal. Do NOT read it as "NC" or add extra characters.
- E vs C: Check if there are horizontal strokes - if yes, it's likely E, not C. Even if a very small horizontal stroke is present, it is likely E, not C.
- Double-check every character before finalizing - especially V, N, E, C, G at word boundaries.

CUSTOMER NAME EXTRACTION - Special Attention:
- Read ALL character boxes in the name field from LEFT to RIGHT, including trailing single letters.
- Many Indian names have initials at the END (e.g., "AMAN DEEP G", "VARUN M", "KUMAR A").
- DO NOT skip or drop single-letter initials at the end of names.
- Preserve SINGLE SPACES between words. Names with middle initials like "MUNEER V V" should have spaces preserved.
- After extraction, verify: Did I capture every filled box? Are there any trailing initials I missed?

Instructions:
- If a field cannot be found, set its value to null.
- The PAN number is always 10 characters: 5 alphabets, 4 digits, 1 alphabet.
- The Barcode Number is typically in top right of the document.

CRITICAL - Barcode Validation:
- Carefully examine the Barcode_Number field for ANY signs of:
  * Cuttings (lines scratched through digits)
  * Crossings (digits crossed out with X or lines)
  * Overwriting (digits written over other digits)
  * Corrections or alterations of any kind
  * Outside of the barcode number box
- If the Barcode_Number has ANY cuttings or overwriting or outside of the barcode number box:
  * Set "Barcode_Number" value to null
  * Set "Barcode_Formatting" value to a short reason, e.g., "The barcode number has cuttings/crossings/overwritings/outside of the barcode number box"
- If the Barcode_Number is clean and unaltered:
  * Set "Barcode_Formatting" value to "Valid"

CONFIDENCE SCORES:
For each field, provide a confidence_score between 0.0 and 1.0:
- 1.0 = Absolutely certain, clearly visible and readable
- 0.9-0.95 = High confidence, minor ambiguity
- 0.5-0.8 = Medium confidence, some characters unclear or partially visible
- 0.1-0.4 = Low confidence, mostly guessing based on context
- 0.0 = Field not found (value is null)

Return ONLY a valid JSON object in this EXACT nested format:
(Do NOT wrap the JSON in ```json or any other formatting.)
{
  "Customer_Name": {
    "value": "<extracted value or null>",
    "confidence_score": <0.0 to 1.0>
  },
  "PAN_Number": {
    "value": "<extracted value or null>",
    "confidence_score": <0.0 to 1.0>
  },
  "Barcode_Number": {
    "value": "<extracted value or null>",
    "confidence_score": <0.0 to 1.0>
  },
  "Barcode_Formatting": {
    "value": "<Valid or reason for invalidity>",
    "confidence_score": <0.0 to 1.0>
  }
}

Do not include any explanations or additional text.
"""


def create_field(value, confidence_score: float = 0.0) -> dict:
    """Create a field object with value and confidence_score."""
    return {"value": value, "confidence_score": confidence_score}


def create_error_response(error_msg: str) -> dict:
    """Create an error response with null values and zero confidence."""
    return {
        "Customer_Name": create_field(None, 0.0),
        "PAN_Number": create_field(None, 0.0),
        "Barcode_Number": create_field(None, 0.0),
        "Barcode_Formatting": create_field(None, 0.0),
        "error": error_msg
    }


@tool("extract_signature_card_data")
def extract_signature_card_data(image_path: str) -> str:
    """
    Extract Customer Name, PAN Number, Barcode Number and Barcode Formatting from a signature card image,
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
        
        # Apply preprocessing to enhance handwritten text (COMMENTED OUT FOR NOW)
        # image = enhance_handwritten_text(
        #     image,
        #     apply_clahe_enhancement=True,
        #     clahe_clip_limit=2.5,
        #     apply_sharpening_enhancement=True,
        #     sharpening_strength=1.0,
        #     apply_denoising_enhancement=False,  # Can enable if images are noisy
        #     brightness=10,
        #     contrast=1.1
        # )
        
        # Initialize model with generation config
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
        )
        model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)
        
        # Generate response
        response = model.generate_content([EXTRACTION_PROMPT, image])
        
        # Clean up response
        raw = response.text.strip() if response.text else "{}"
        clean = raw.replace("```json", "").replace("```", "").strip()
        
        try:
            # Validate JSON
            parsed = json.loads(clean)
            
            # Ensure all fields have the nested format with value and confidence_score
            for field in ["Customer_Name", "PAN_Number", "Barcode_Number", "Barcode_Formatting"]:
                if field in parsed:
                    # If field is not in nested format, convert it
                    if not isinstance(parsed[field], dict):
                        parsed[field] = create_field(parsed[field], 0.5)
                    else:
                        # Ensure both keys exist
                        if "value" not in parsed[field]:
                            parsed[field]["value"] = None
                        if "confidence_score" not in parsed[field]:
                            parsed[field]["confidence_score"] = 0.0 if parsed[field]["value"] is None else 0.5
                else:
                    parsed[field] = create_field(None, 0.0)
            
            return json.dumps(parsed)
        except json.JSONDecodeError:
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

