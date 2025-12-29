"""
Image Preprocessing Tool.
Applies image enhancements based on quality assessment recommendations.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
import tempfile
from crewai.tools import tool


def adjust_brightness(image: np.ndarray, adjustment: float) -> np.ndarray:
    """Adjust image brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Adjust value channel
    adjustment = int(adjustment)
    if adjustment > 0:
        v = cv2.add(v, adjustment)
    else:
        v = cv2.subtract(v, abs(adjustment))
    
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust image contrast using CLAHE."""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE with clip limit based on factor
    clip_limit = min(4.0, max(1.0, factor))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    # Merge back
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def apply_sharpening(image: np.ndarray, strength: float) -> np.ndarray:
    """Apply unsharp masking to sharpen the image."""
    strength = min(2.5, max(0.5, strength))
    gaussian = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
    return sharpened


def apply_denoising(image: np.ndarray, strength: int) -> np.ndarray:
    """Apply denoising filter."""
    strength = min(15, max(3, strength))
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def deskew_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image to correct skew."""
    if abs(angle) < 0.5:
        return image
    
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box size
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # Adjust rotation matrix
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def upscale_image(image: np.ndarray, factor: float) -> np.ndarray:
    """Upscale image resolution."""
    factor = min(2.0, max(1.0, factor))
    if factor <= 1.0:
        return image
    
    height, width = image.shape[:2]
    new_width = int(width * factor)
    new_height = int(height * factor)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


@tool("preprocess_image")
def preprocess_image(image_path: str, adjustments_json: str) -> str:
    """
    Apply preprocessing to an image based on recommended adjustments.
    
    Args:
        image_path: Path to the image file to preprocess
        adjustments_json: JSON string containing adjustments to apply:
            - brightness: int (positive to brighten, negative to darken)
            - contrast: float (CLAHE clip limit factor)
            - sharpening: float (sharpening strength, 0.5-2.5)
            - denoising: int (denoising strength, 3-15)
            - deskew: float (angle to rotate in degrees)
            - upscale: float (upscale factor, 1.0-2.0)
    
    Returns:
        str: JSON string with the path to the preprocessed image and applied adjustments
    """
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            return json.dumps({
                "error": f"Image file not found: {image_path}",
                "success": False
            })
        
        # Parse adjustments
        try:
            adjustments = json.loads(adjustments_json)
        except json.JSONDecodeError:
            adjustments = {}
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return json.dumps({
                "error": f"Failed to load image: {image_path}",
                "success": False
            })
        
        applied_adjustments = []
        
        # Apply deskew first (before other transformations)
        if "deskew" in adjustments and abs(adjustments["deskew"]) > 0.5:
            image = deskew_image(image, adjustments["deskew"])
            applied_adjustments.append(f"Deskewed by {adjustments['deskew']:.1f} degrees")
        
        # Apply upscaling
        if "upscale" in adjustments and adjustments["upscale"] > 1.0:
            image = upscale_image(image, adjustments["upscale"])
            applied_adjustments.append(f"Upscaled by {adjustments['upscale']:.1f}x")
        
        # Apply denoising (before sharpening to avoid amplifying noise)
        if "denoising" in adjustments and adjustments["denoising"] > 0:
            image = apply_denoising(image, int(adjustments["denoising"]))
            applied_adjustments.append(f"Denoised with strength {adjustments['denoising']}")
        
        # Apply brightness adjustment
        if "brightness" in adjustments and abs(adjustments["brightness"]) > 5:
            image = adjust_brightness(image, adjustments["brightness"])
            applied_adjustments.append(f"Brightness adjusted by {adjustments['brightness']:.0f}")
        
        # Apply contrast enhancement
        if "contrast" in adjustments and adjustments["contrast"] > 1.0:
            image = adjust_contrast(image, adjustments["contrast"])
            applied_adjustments.append(f"Contrast enhanced with factor {adjustments['contrast']:.1f}")
        
        # Apply sharpening last
        if "sharpening" in adjustments and adjustments["sharpening"] > 0:
            image = apply_sharpening(image, adjustments["sharpening"])
            applied_adjustments.append(f"Sharpened with strength {adjustments['sharpening']:.1f}")
        
        # Save preprocessed image to a new temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='_preprocessed.png') as tmp_file:
            output_path = tmp_file.name
        
        cv2.imwrite(output_path, image)
        
        result = {
            "success": True,
            "preprocessed_image_path": output_path,
            "applied_adjustments": applied_adjustments,
            "message": f"Successfully applied {len(applied_adjustments)} preprocessing steps"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "success": False
        })

