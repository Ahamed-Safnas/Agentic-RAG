import logfire
from nemoguardrails import RailsConfig, LLMRails

from app.gateway import get_langchain_llm
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS
from app.config import settings

_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup using the Portkey LLM Gateway.
    """
    global _rails

    guard_llm = get_langchain_llm(
        feature="guardrails", 
        model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised via Portkey Gateway.")
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        # Defensively strip any reasoning <think> tags if present
        import re
        clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        fired = any(indicator.lower() in clean_content.lower() for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, clean_content

        logfire.info("✅ Guardrails passed.")
        return False, None
