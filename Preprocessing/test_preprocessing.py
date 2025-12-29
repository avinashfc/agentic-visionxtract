"""
Test script for the Preprocessing Pipeline.

Usage:
    python test_preprocessing.py                     # Uses a sample test image
    python test_preprocessing.py path/to/image.jpg   # Uses specified image
"""

import sys
import os
from pathlib import Path
from PIL import Image

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from crewai_workflow.app import PreprocessingCrew, process_image_for_ocr


def test_with_image(image_path: str):
    """Test preprocessing pipeline with a specific image."""
    print("\n" + "=" * 70)
    print("PREPROCESSING PIPELINE TEST")
    print("=" * 70)
    print(f"\nInput Image: {image_path}")
    
    # Load image
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        print(f"Image Size: {image.size}")
        print(f"Image Mode: {image.mode}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Process through pipeline
    print("\nStarting preprocessing pipeline...")
    result = process_image_for_ocr(image)
    
    # Display results
    print("\n" + "=" * 70)
    print("PIPELINE RESULT")
    print("=" * 70)
    print(f"\nStatus: {result.get('status', 'unknown')}")
    print(f"Message: {result.get('message', 'No message')}")
    
    if result.get('quality_scores'):
        print("\nFinal Quality Scores:")
        for key, value in result['quality_scores'].items():
            print(f"  {key}: {value}")
    
    if result.get('preprocessing_applied'):
        print("\nPreprocessing was applied:")
        for adj in result.get('adjustments_applied', []):
            print(f"  - {adj}")
    
    if result.get('remaining_issues'):
        print("\nRemaining Issues:")
        for issue in result['remaining_issues']:
            print(f"  - {issue}")
    
    if result.get('image_path'):
        print(f"\nOutput Image: {result['image_path']}")
    
    print("\n" + "=" * 70)
    
    # Close image
    image.close()
    
    return result


def create_test_image():
    """Create a simple test image with quality issues."""
    import numpy as np
    
    # Create a slightly dark, low contrast image
    width, height = 800, 600
    img_array = np.ones((height, width, 3), dtype=np.uint8) * 100  # Dark gray
    
    # Add some text-like patterns (simulated)
    for i in range(10):
        y = 50 + i * 50
        img_array[y:y+20, 100:700] = 60  # Darker lines
    
    # Add some noise
    noise = np.random.normal(0, 10, (height, width, 3))
    img_array = np.clip(img_array.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    test_image = Image.fromarray(img_array)
    
    # Save temporarily
    test_path = "/tmp/test_preprocessing_sample.png"
    test_image.save(test_path)
    print(f"Created test image: {test_path}")
    
    return test_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use provided image path
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)
    else:
        # Create a test image
        print("No image provided. Creating a test image with quality issues...")
        image_path = create_test_image()
    
    test_with_image(image_path)

