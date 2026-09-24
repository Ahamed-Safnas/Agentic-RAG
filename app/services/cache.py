"""Best-effort Redis caching for exact and semantic responses."""

import hashlib
import json
import math
from typing import Any

import logfire
from redis import Redis

from app.config import settings

_client: Redis | None = None


def _get_client() -> Redis:
    global _client
    if _client is None:
        _client = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
    return _client


def _key(namespace: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"rag:{namespace}:{settings.CACHE_VERSION}:{digest}"


def normalize_query(query: str) -> str:
    """Normalize whitespace and casing so equivalent questions share a key."""
    return " ".join(query.casefold().split())


def make_response_key(query: str, thread_id: str | None) -> str:
    return _key(
        "answer",
        thread_id or "default_user",
        settings.QDRANT_COLLECTION,
        settings.PORTKEY_RESPONDER_MODEL,
        normalize_query(query),
    )


def _answer_index_key() -> str:
    return f"rag:answer-index:{settings.CACHE_VERSION}"


def _answer_metadata_key(response_key: str) -> str:
    return f"{response_key}:semantic"


def _get_json(key: str) -> dict[str, Any] | None:
    try:
        value = _get_client().get(key)
        return json.loads(value) if value else None
    except Exception as exc:
        logfire.warning(f"Redis cache read failed: {exc}")
        return None


def _set_json(key: str, value: dict[str, Any], ttl_seconds: int) -> None:
    try:
        _get_client().setex(key, ttl_seconds, json.dumps(value, ensure_ascii=True))
    except Exception as exc:
        logfire.warning(f"Redis cache write failed: {exc}")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def get_exact_cached_response(query: str, thread_id: str | None) -> dict[str, Any] | None:
    """Return an exact response-cache match without creating an embedding."""
    response_key = make_response_key(query, thread_id)
    exact_response = _get_json(response_key)
    return exact_response if isinstance(exact_response, dict) else None


def get_semantic_cached_response(
    query: str,
    thread_id: str | None,
    query_embedding: list[float],
) -> dict[str, Any] | None:
    """Return a semantically equivalent response using a supplied embedding."""

    try:
        client = _get_client()
        metadata_keys = list(client.smembers(_answer_index_key()))[-settings.SEMANTIC_CACHE_MAX_ENTRIES :]
        best_match: tuple[float, dict[str, Any]] | None = None

        for metadata_key in metadata_keys:
            raw_metadata = client.get(metadata_key)
            if not raw_metadata:
                continue
            metadata = json.loads(raw_metadata)
            if metadata.get("thread_id") != (thread_id or "default_user"):
                continue

            similarity = _cosine_similarity(query_embedding, metadata.get("embedding", []))
            if similarity < settings.SEMANTIC_CACHE_THRESHOLD:
                continue

            cached_response = _get_json(metadata["response_key"])
            if isinstance(cached_response, dict) and (best_match is None or similarity > best_match[0]):
                best_match = (similarity, cached_response)

        if best_match:
            logfire.info(f"✅ Semantic response cache hit (similarity={best_match[0]:.3f})")
            return best_match[1]
    except Exception as exc:
        logfire.warning(f"Semantic cache lookup failed: {exc}")

    return None


def set_cached_response(
    query: str,
    thread_id: str | None,
    response: dict[str, Any],
    ttl_seconds: int,
    query_embedding: list[float] | None = None,
) -> None:
    """Store a response and its supplied embedding for future paraphrase matches."""
    response_key = make_response_key(query, thread_id)
    _set_json(response_key, response, ttl_seconds)

    if query_embedding is None:
        return

    try:
        metadata_key = _answer_metadata_key(response_key)
        metadata = {
            "response_key": response_key,
            "thread_id": thread_id or "default_user",
            "embedding": query_embedding,
        }
        client = _get_client()
        client.setex(metadata_key, ttl_seconds, json.dumps(metadata, ensure_ascii=True))
        client.sadd(_answer_index_key(), metadata_key)
        client.expire(_answer_index_key(), ttl_seconds)
    except Exception as exc:
        logfire.warning(f"Semantic cache write failed: {exc}")