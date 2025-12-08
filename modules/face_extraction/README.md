# Face Extraction Module

This module provides face extraction capabilities from uploaded documents using Google ADK and Vision API.

## Face Extraction Pipeline

The face extraction pipeline uses **Google Agent Development Kit (ADK)** and Google Vision API to detect and extract faces from uploaded documents.

### Architecture

The pipeline is built using Google ADK's framework:
- **ADK Agent** (`agents/face_extraction_agent.py`): Main agent orchestrating the face extraction process
- **ADK Tools** (`tools/face_extraction_tools.py`): Wraps functionality as ADK FunctionTools for agent use
- **Underlying Tools**: 
  - `document_uploader.py`: Handles document upload and validation
  - `face_detector.py`: Detects and extracts faces using Google Vision API
- **Workflow** (`workflows/face_extraction_workflow.py`): Orchestrates the complete pipeline using ADK
- **Models** (`models/face_extraction.py`): Pydantic models for request/response
- **Router** (`routers/face_extraction.py`): FastAPI endpoints for face extraction

### ADK Integration

The pipeline uses Google ADK's structured approach:

1. **ADK Agent**: Created using `google.adk.Agent` with tools
2. **ADK Tools**: Functions wrapped as `FunctionTool` instances for agent use
3. **Tool Functions**:
   - `validate_document`: Validates uploaded images
   - `upload_document`: Stores documents (local or GCS)
   - `detect_faces`: Detects faces using Vision API
   - `extract_face_images`: Extracts face crops from images

The agent can optionally use Gemini models for enhanced reasoning, but face detection works independently using Vision API.

### API Endpoints

#### Extract Faces from Document
```bash
POST /api/face-extraction/extract-faces
Content-Type: multipart/form-data

Parameters:
- file: Document image file (required)
- min_confidence: Minimum confidence threshold (default: 0.7)
- extract_all_faces: Extract all faces or just first (default: true)
```

#### Batch Face Extraction
```bash
POST /api/face-extraction/extract-faces-batch
Content-Type: multipart/form-data

Parameters:
- files: List of document image files (required)
- min_confidence: Minimum confidence threshold (default: 0.7)
- extract_all_faces: Extract all faces or just first (default: true)
```

### Usage Example

```python
import requests

# Single document
with open('document.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/api/face-extraction/extract-faces',
        files=files,
        params={'min_confidence': 0.7}
    )
    result = response.json()
    print(f"Found {result['faces_detected']} faces")
```

### Configuration

#### Required: Google Cloud Credentials

The Vision API requires service account credentials. Set up one of the following:

**Option 1: Service Account JSON File (Recommended for production)**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

**Option 2: Application Default Credentials (For local development)**
```bash
gcloud auth application-default login
```

**Option 3: Set in .env file**
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

#### Optional Environment Variables

```bash
GOOGLE_API_KEY=your-api-key  # For future Gemini features (not required for face detection)
GCS_BUCKET_NAME=your-bucket-name  # Optional, for document storage
GEMINI_MODEL_NAME=gemini-2.0-flash-exp  # Optional
```

**Note:** Vision API requires service account credentials, not an API key. The `GOOGLE_API_KEY` is reserved for future Gemini model features.

### Requirements

- Google Cloud Vision API enabled
- Valid Google Cloud credentials or API key
- Pillow for image processing
- google-cloud-vision for face detection

