"""
Judge evaluation helper for face extraction responses.
"""
from typing import Optional, List, Dict, Any
from core.module_client import ModuleClient, CommunicationMode
from modules.face_extraction.models.face_extraction import FaceExtractionResponse
import logging

logger = logging.getLogger(__name__)


async def evaluate_with_judge(
    response: FaceExtractionResponse,
    document_name: str,
    min_confidence: float,
    judge_criteria: Optional[List[Dict[str, Any]]],
    judge_task_description: Optional[str],
) -> FaceExtractionResponse:
    """Always evaluate with LLM Judge via A2A when called."""
    if not response.faces_extracted:
        return response

    try:
        # Prepare content for evaluation
        faces_info = f"Document: {document_name}\n"
        faces_info += f"Faces detected: {response.faces_detected}\n"
        faces_info += f"Min confidence threshold: {min_confidence}\n"
        for idx, face in enumerate(response.faces_extracted, 1):
            faces_info += f"Face {idx}: ID={face.face_id}, Confidence={face.bounding_box.confidence}\n"

        async with ModuleClient("llm_judge", mode=CommunicationMode.AUTO) as judge_client:
            evaluation = await judge_client.evaluate(
                content=faces_info,
                criteria=judge_criteria,
                task_description=judge_task_description or f"Evaluate face extraction quality for document: {document_name}",
                context={
                    "document_name": document_name,
                    "faces_detected": response.faces_detected,
                    "min_confidence": min_confidence,
                },
            )

            if not hasattr(response, "metadata") or response.metadata is None:
                response.metadata = {}
            response.metadata["evaluation"] = evaluation
            response.metadata["evaluated"] = True
    except Exception as e:
        logger.warning(f"Failed to evaluate face extraction result with judge: {e}")
        if not hasattr(response, "metadata") or response.metadata is None:
            response.metadata = {}
        response.metadata["evaluation_error"] = str(e)
        response.metadata["evaluated"] = False

    return response

