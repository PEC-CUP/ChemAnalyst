from typing import Final

from app.utils.logger import setup_logger


logger = setup_logger(__name__)

RAG_KEYWORDS: Final[tuple[str, ...]] = (
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
)
TOOL_KEYWORDS: Final[tuple[str, ...]] = (
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "OCR",
    "ocr",
    "text",
    "text",
    "text",
    "text",
    "textPDF",
    "textpdf",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "GC",
    "gc",
)
REASONING_KEYWORDS: Final[tuple[str, ...]] = ("text", "text", "text", "text", "text", "text", "text")
TASK_TYPES: Final[set[str]] = {"chat", "rag", "tool", "reasoning"}


def _is_document_parse_request(normalized: str) -> bool:
    action_words = ("text", "text", "text", "text")
    file_words = ("text", "text", "text", "text", "text", "text", "text", "PDF", "pdf", "text")
    return any(action in normalized for action in action_words) and any(file_word in normalized for file_word in file_words)


def detect_task_type(query: str) -> str:
    """Simple rule-based router for v1."""
    normalized = query.strip()

    if any(keyword in normalized for keyword in RAG_KEYWORDS):
        task_type = "rag"
    elif _is_document_parse_request(normalized):
        task_type = "tool"
    elif any(keyword in normalized for keyword in TOOL_KEYWORDS):
        task_type = "tool"
    elif any(keyword in normalized for keyword in REASONING_KEYWORDS):
        task_type = "reasoning"
    else:
        task_type = "chat"

    logger.info("Task routed to: %s", task_type)
    return task_type


def normalize_task_type(task_type: str | None) -> str | None:
    """Validate frontend or API requested task type override."""
    if task_type is None:
        return None

    normalized = task_type.strip().lower()
    if not normalized or normalized == "auto":
        return None
    if normalized not in TASK_TYPES:
        raise ValueError(f"text task_type: {task_type}")
    return normalized
