# Agent-to-Agent (A2A) Communication Pattern

This document describes the A2A communication pattern used for inter-module communication in the Agentic VisionXtract system.

## Overview

The A2A pattern enables modules to communicate with each other in a modular, scalable way. It supports both:
- **In-process mode**: Direct Python function calls (unified deployment)
- **HTTP mode**: REST API calls (distributed/microservices deployment)

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Module A  │ ──A2A──> │  Module      │ ──A2A──> │   Module B  │
│   (OCR)     │         │  Client      │         │   (Judge)   │
└─────────────┘         │  (Adapter)   │         └─────────────┘
                        └──────────────┘
                              │
                              │ Supports:
                              ├─ In-process (direct import)
                              └─ HTTP (distributed service)
```

## ModuleClient

The `ModuleClient` class in `core/module_client.py` provides a unified interface for A2A communication.

### Features

- **Auto-detection**: Automatically detects whether to use in-process or HTTP mode
- **Unified API**: Same code works for both deployment modes
- **Error handling**: Graceful fallback and error handling
- **Type safety**: Returns properly typed responses

### Usage Example

```python
from core.module_client import ModuleClient, CommunicationMode

# Auto-detect mode (recommended)
async with ModuleClient("llm_judge", mode=CommunicationMode.AUTO) as judge_client:
    evaluation = await judge_client.evaluate(
        content="Text to evaluate",
        task_description="Evaluate OCR quality"
    )

# Explicit in-process mode
async with ModuleClient("llm_judge", mode=CommunicationMode.IN_PROCESS) as judge_client:
    result = await judge_client.evaluate(...)

# Explicit HTTP mode
async with ModuleClient("llm_judge", mode=CommunicationMode.HTTP, base_url="http://localhost:8003") as judge_client:
    result = await judge_client.evaluate(...)
```

## Configuration

### Environment Variables

For HTTP mode, you can configure module URLs:

```bash
# Set module URL for distributed deployment
MODULE_LLM_JUDGE_URL=http://judge-service:8003
MODULE_OCR_URL=http://ocr-service:8002
```

If not set, defaults to `http://localhost:{port}` where port is determined by module name.

### Mode Detection

The `AUTO` mode detects the appropriate communication method:
1. Checks for `MODULE_{NAME}_URL` environment variable (HTTP mode)
2. Tries to import module in-process (IN_PROCESS mode)
3. Falls back to HTTP mode if import fails

## Example: OCR with Judge Evaluation

The OCR module uses the judge module via A2A to evaluate OCR results:

```python
# In OCR workflow
async with ModuleClient("llm_judge", mode=CommunicationMode.AUTO) as judge_client:
    evaluation = await judge_client.evaluate(
        content=ocr_result.full_text,
        task_description="Evaluate OCR extraction quality",
        context={"document_name": document_name}
    )
    
    # Add evaluation to response metadata
    response.metadata['evaluation'] = evaluation
```

### API Endpoints

**OCR with evaluation:**
```bash
# Extract text with optional evaluation
POST /api/ocr/extract-text?evaluate_with_judge=true

# Extract and always evaluate
POST /api/ocr/extract-and-evaluate
```

## Benefits

1. **Modularity**: Modules remain independent and reusable
2. **Scalability**: Can deploy modules separately and scale independently
3. **Flexibility**: Same code works in unified or distributed deployments
4. **Testability**: Easy to mock modules for testing
5. **Maintainability**: Changes to one module don't affect others

## Adding A2A to Your Module

To add A2A communication to your module:

1. **Import ModuleClient:**
   ```python
   from core.module_client import ModuleClient, CommunicationMode
   ```

2. **Use in your workflow:**
   ```python
   async with ModuleClient("target_module", mode=CommunicationMode.AUTO) as client:
       result = await client.call("method_name", payload={...})
   ```

3. **For judge module specifically:**
   ```python
   async with ModuleClient("llm_judge", mode=CommunicationMode.AUTO) as judge_client:
       evaluation = await judge_client.evaluate(...)
       comparison = await judge_client.compare(...)
   ```

## Best Practices

1. **Use AUTO mode**: Let the system detect the appropriate mode
2. **Handle errors gracefully**: A2A calls may fail - don't let them break your module
3. **Use async context managers**: Always use `async with` for proper resource cleanup
4. **Make evaluation optional**: Don't require judge evaluation for core functionality
5. **Log A2A calls**: Helpful for debugging distributed deployments

## Current Implementation

- ✅ **OCR Module**: Uses judge for optional evaluation
- ✅ **Judge Module**: Available for evaluation and comparison
- ✅ **ModuleClient**: Supports in-process and HTTP modes
- ✅ **Auto-discovery**: Automatically detects communication mode

## Future Enhancements

- [ ] Add retry logic for HTTP calls
- [ ] Add circuit breaker pattern
- [ ] Add metrics/monitoring for A2A calls
- [ ] Support for message queues (RabbitMQ, Kafka)
- [ ] Add authentication for HTTP mode

