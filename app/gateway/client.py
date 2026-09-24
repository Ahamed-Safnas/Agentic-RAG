from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.config import settings

# Portkey routing strategy:
#   - Primary/fallback logic lives in a Portkey saved config (required when
#     block_inline_config is enabled on the workspace).
#   - We reference that config via the x-portkey-config-id header.
#   - The inline config dict approach is disabled for this account, so all
#     retry/fallback behavior must be configured inside the Portkey UI.


def _make_headers(feature: str = "rag") -> dict:
    """Build Portkey headers that reference the primary saved config by ID."""
    config_ids = {
        "planner": settings.PORTKEY_PLANNER_CONFIG_ID,
        "responder": settings.PORTKEY_RESPONDER_CONFIG_ID,
        "guardrails": settings.PORTKEY_GUARDRAILS_CONFIG_ID,
        "evals": settings.PORTKEY_EVALS_CONFIG_ID,
    }
    config_id = config_ids.get(feature) or settings.PORTKEY_PRIMARY_CONFIG_ID

    if not config_id:
        raise ValueError(
            "PORTKEY_PRIMARY_CONFIG_ID is not set in .env. "
            "Get the real pc-... ID from the Portkey dashboard or "
            "run: PYTHONPATH=. python scripts/list_portkey_configs.py"
        )
    return createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        config_id=config_id,
        metadata={
            "feature": feature,
            "_user": "rag-system",
            "environment": "production",
        },
    )


def get_portkey_model(feature: str) -> str:
    model = {
        "planner": settings.PORTKEY_PLANNER_MODEL,
        "responder": settings.PORTKEY_RESPONDER_MODEL,
        "guardrails": settings.PORTKEY_GUARDRAILS_MODEL,
        "evals": settings.PORTKEY_EVALS_MODEL,
    }.get(feature, settings.PORTKEY_RESPONDER_MODEL)
    if model.startswith("@"):
        return model
    return f"@{settings.PORTKEY_PRIMARY_SLUG}/{model}"


# OpenAI-compatible client routed through Portkey.
# We use the OpenAI SDK directly because the native Portkey SDK does not
# surface a first-class config_id constructor parameter; the header-based
# approach works reliably with block_inline_config enabled.
portkey_client = OpenAI(
    api_key=settings.PORTKEY_API_KEY,
    base_url=PORTKEY_GATEWAY_URL,
    default_headers=_make_headers(),
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI - a drop-in for LangChain nodes.

    Why ChatOpenAI:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + saved-config reference). The @slug/model-name format is Portkey-specific - the
      upstream provider's own client does not understand it. Portkey is just in the middle.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=get_portkey_model(feature),
        default_headers=_make_headers(feature),
    )


def get_async_openai_client(feature: str = "rag") -> AsyncOpenAI:
    """
    Returns an async OpenAI client that routes through the Portkey gateway.
    Use this for non-LangChain async LLM calls (e.g. async FastAPI endpoints).
    """
    return AsyncOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=_make_headers(feature),
    )


