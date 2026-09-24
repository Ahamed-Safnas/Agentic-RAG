import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings

# 1. Store the Config ID string instead of an inline dictionary
# (You can also add PORTKEY_CONFIG_ID to your settings.py)
# PORTKEY_CONFIG_ID = settings.PORTKEY_CONFIG_ID  # e.g., "pc-my-fallback-config-1234"

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=settings.PORTKEY_CONFIG_ID  # 👈 Pass the string ID
)


def get_langchain_llm(
    model: str,
    temperature: float = 0,
    feature: str = "rag",
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=model,
        temperature=temperature,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=settings.PORTKEY_CONFIG_ID,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production",
            },
        ),
    )
def extract_cache_status(response) -> str:
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"