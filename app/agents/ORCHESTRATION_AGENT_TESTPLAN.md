# Orchestration Agent Test Plan

This file records focused checks for the ChemAnalyst orchestration layer.

## Scope

- Task classification into simple and complex requests.
- Upload-required workflows and missing-file responses.
- Tool dispatch for molecular-composition quantification, GC matching, document parsing, and property inference.
- RAG evidence retrieval and answer grounding.
- Error handling when an LLM endpoint or tool backend is unavailable.

## Expected Behavior

The planner should avoid unsupported inferences, expose clear failure messages, and keep raw tool outputs available for debugging while presenting simplified results to non-expert users.
