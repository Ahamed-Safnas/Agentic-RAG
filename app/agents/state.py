from typing import Annotated, List, NotRequired, TypedDict

from app.config import settings


def merge_bounded_messages(existing: List[dict], incoming: List[dict]) -> List[dict]:
    """Append messages while enforcing bounded persisted conversation memory."""
    messages = (existing + incoming)[-settings.MEMORY_MAX_MESSAGES :]

    retained: list[dict] = []
    retained_chars = 0
    for message in reversed(messages):
        message_chars = len(str(message.get("content", "")))
        if retained and retained_chars + message_chars > settings.MEMORY_MAX_CHARS:
            break
        retained.append(message)
        retained_chars += message_chars

    return list(reversed(retained))


class AgentState(TypedDict):
    messages: Annotated[List[dict], merge_bounded_messages]
    current_query: str
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str
    query_embedding: NotRequired[list[float] | None]
