# Runtime and Orchestration Architecture

ChemAnalyst uses a lightweight runtime layer for direct interactions and a planner layer for multi-step analytical workflows.

## Entry Points

- `app/agents/task_classifier.py` classifies incoming requests into simple or complex tasks.
- `app/agents/runtime_agent.py` handles direct chat, RAG calls, upload continuation, recent-result follow-up, and compatibility behavior.
- `app/agents/planner_orchestrator.py` coordinates multi-step workflows that combine uploaded files, tools, retrieval evidence, and final reasoning.

## Runtime Agent

`runtime_agent.py` is the user-facing execution wrapper. It is responsible for simple Q&A, direct RAG routing, file-upload state, tool execution, and follow-up questions over recent structured results.

It should remain a compact compatibility and interaction layer, not the long-term home for complex analytical orchestration.

## Planner Orchestrator

`planner_orchestrator.py` is the main complex-task coordinator. It builds workflow plans, dispatches tools, checks evidence, and produces final grounded answers.

Typical coordinated capabilities include document parsing, schema checks, GC matching, quantitative tool execution, petroleum knowledge-base retrieval, historical experimental-database evidence retrieval, evidence sufficiency checks, and final synthesis with explicit grounding constraints.

## Recommended Route

Simple requests stay in `runtime_agent.py`. Complex analytical requests enter `planner_orchestrator.py` after task classification. The design keeps one front-door classifier and prevents multiple routing systems from competing for the same responsibility.
