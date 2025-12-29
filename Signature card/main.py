from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import re
from crewai_workflow.app import SignatureCardCrew

# Initialize FastAPI app
app = FastAPI(title="Signature Card Processing API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_crewai_json_result(result_str: str) -> dict:
    """Parse CrewAI result string to JSON, handling various formats."""
    if not result_str:
        return {}
    
    # If already a dict, return it
    if isinstance(result_str, dict):
        return result_str
    
    # Clean up the string
    result_str = str(result_str).strip()
    
    # Remove markdown code blocks if present
    result_str = re.sub(r'^```json\s*', '', result_str)
    result_str = re.sub(r'^```\s*', '', result_str)
    result_str = re.sub(r'\s*```$', '', result_str)
    result_str = result_str.strip()
    
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        # Try to find JSON object in the string
        json_match = re.search(r'\{[^{}]*\}', result_str, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Return as raw text if parsing fails
        return {"raw_result": result_str}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Signature Card Processing API is running", "status": "healthy"}


@app.post("/process-signature-card")
async def process_signature_card_endpoint(file: UploadFile = File(...)):
    """
    Process a signature card image or PDF to extract:
    - Customer Name
    - PAN number
    - Barcode Number
    - Stamp presence (true/false)
    
    Supports: JPEG, PNG, WebP images and PDF files
    """
    try:
        # Validate file type
        if not is_supported_file(file.content_type, file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (JPEG, PNG, WebP) or PDF"
            )
        
        # Read file contents
        contents = await file.read()
        
        # Convert file to PIL image (handles both PDFs and images)
        pil_image = convert_file_to_image(contents, file.content_type, file.filename)
        
        if pil_image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process the file. Please ensure it's a valid image or PDF."
            )
        
        # Process signature card using CrewAI
        signature_card_crew = SignatureCardCrew.get_instance()
        result = signature_card_crew.process_signature_card(pil_image)
        
        # Clean up image from memory immediately after processing
        pil_image.close()
        del contents
        del pil_image

        # Handle different result types from CrewAI
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="CrewAI returned None result"
            )
        
        if isinstance(result, dict):
            parsed_result = result
        elif isinstance(result, str):
            parsed_result = parse_crewai_json_result(result)
        else:
            result_str = str(result) if result else "{}"
            parsed_result = parse_crewai_json_result(result_str)

        print(f"Processed result: {parsed_result}")

        return {
            "result": parsed_result,
            "filename": file.filename,
            "file_type": "pdf" if is_pdf(file.content_type, file.filename) else "image"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


@app.post("/extract-fields")
async def extract_fields_only(file: UploadFile = File(...)):
    """
    Extract only Name, PAN, and Barcode from signature card (without stamp detection).
    Useful for faster processing when stamp detection is not needed.
    
    Supports: JPEG, PNG, WebP images and PDF files
    """
    try:
        # Validate file type
        if not is_supported_file(file.content_type, file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (JPEG, PNG, WebP) or PDF"
            )
        
        # Read file contents
        contents = await file.read()
        
        # Convert file to PIL image (handles both PDFs and images)
        pil_image = convert_file_to_image(contents, file.content_type, file.filename)
        
        if pil_image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process the file. Please ensure it's a valid image or PDF."
            )
        
        # Use the extraction tool directly
        from crewai_workflow.agents.extractor_agent.tools.extract_data import extract_signature_card_data
        
        # Save image temporarily
        temp_path = save_image_to_temp(pil_image)
        
        try:
            # Call the extraction tool directly
            result = extract_signature_card_data(temp_path)
            parsed_result = json.loads(result) if isinstance(result, str) else result
        finally:
            # Clean up temp file
            cleanup_temp_file(temp_path)
        
        pil_image.close()
        del contents
        del pil_image

        return {
            "result": parsed_result,
            "filename": file.filename,
            "file_type": "pdf" if is_pdf(file.content_type, file.filename) else "image"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}"
        )


@app.post("/detect-stamp")
async def detect_stamp_only(file: UploadFile = File(...)):
    """
    Detect only stamp presence on signature card (without field extraction).
    Useful when only stamp verification is needed.
    
    Supports: JPEG, PNG, WebP images and PDF files
    """
    try:
        # Validate file type
        if not is_supported_file(file.content_type, file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (JPEG, PNG, WebP) or PDF"
            )
        
        # Read file contents
        contents = await file.read()
        
        # Convert file to PIL image (handles both PDFs and images)
        pil_image = convert_file_to_image(contents, file.content_type, file.filename)
        
        if pil_image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process the file. Please ensure it's a valid image or PDF."
            )
        
        # Use the stamp detection tool directly
        from crewai_workflow.agents.stamp_detector_agent.tools.detect_stamp import detect_stamp
        
        # Save image temporarily
        temp_path = save_image_to_temp(pil_image)
        
        try:
            # Call the stamp detection tool directly
            result = detect_stamp(temp_path)
            parsed_result = json.loads(result) if isinstance(result, str) else result
        finally:
            # Clean up temp file
            cleanup_temp_file(temp_path)
        
        pil_image.close()
        del contents
        del pil_image

        return {
            "result": parsed_result,
            "filename": file.filename,
            "file_type": "pdf" if is_pdf(file.content_type, file.filename) else "image"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stamp detection failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
