"""
Helper utilities for building OCR responses from agent/tool context.
"""
import time
from typing import Dict
from modules.ocr.models.ocr import KeyValueResponse, KeyValuePair


def build_kv_response_from_context(ctx: Dict, start_time: float) -> KeyValueResponse:
    """Build KeyValueResponse from context."""
    key_value_pairs_raw = ctx.get("key_value_pairs", [])
    full_text = ctx.get("full_text", "")

    # Convert to KeyValuePair objects
    key_value_pairs = []
    for kv in key_value_pairs_raw:
        if isinstance(kv, dict):
            # Ensure key and value are strings (handle None values)
            key = kv.get("key") or ""
            value = kv.get("value")
            if value is None:
                value = ""
            elif not isinstance(value, str):
                value = str(value)
            
            # Skip if both key and value are empty
            if not key and not value:
                continue
                
            key_value_pairs.append(
                KeyValuePair(
                    key=key,
                    value=value,
                    confidence=kv.get("confidence"),
                )
            )

    return KeyValueResponse(
        document_id=ctx.get("document_id", ""),
        key_value_pairs=key_value_pairs,
        raw_text=full_text,
        processing_time=time.time() - start_time,
        status="success",
    )

