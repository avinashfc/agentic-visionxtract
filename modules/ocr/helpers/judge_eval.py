"""
Judge evaluation helper for OCR responses.
"""
from typing import Optional, List, Dict, Any
from core.module_client import ModuleClient, CommunicationMode
from modules.ocr.models.ocr import KeyValueResponse
import logging

logger = logging.getLogger(__name__)


async def evaluate_with_judge(
    response: KeyValueResponse,
    document_name: str,
    language_hints: Optional[List[str]],
    judge_criteria: Optional[List[Dict[str, Any]]],
    judge_task_description: Optional[str],
) -> KeyValueResponse:
    """Always evaluate with LLM Judge via A2A when called."""
    if not response.key_value_pairs:
        return response

    try:
        kv_text = "\n".join([f"{kv.key}: {kv.value}" for kv in response.key_value_pairs])

        async with ModuleClient("llm_judge", mode=CommunicationMode.AUTO) as judge_client:
            evaluation = await judge_client.evaluate(
                content=kv_text,
                criteria=judge_criteria,
                task_description=judge_task_description or f"Evaluate key-value extraction quality for document: {document_name}",
                context={
                    "document_name": document_name,
                    "language_hints": language_hints,
                    "key_value_pairs_count": len(response.key_value_pairs),
                    "raw_text_length": len(response.raw_text),
                },
            )

            if not hasattr(response, "metadata") or response.metadata is None:
                response.metadata = {}
            response.metadata["evaluation"] = evaluation
            response.metadata["evaluated"] = True
    except Exception as e:
        logger.warning(f"Failed to evaluate key-value extraction result with judge: {e}")
        if not hasattr(response, "metadata") or response.metadata is None:
            response.metadata = {}
        response.metadata["evaluation_error"] = str(e)
        response.metadata["evaluated"] = False

    return response

