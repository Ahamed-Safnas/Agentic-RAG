# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire

from app.config import settings

# Logfire v2 EU tokens start with "pylf_v2_eu_" and must send spans to the
# EU endpoint. If no base URL is configured, infer it from the token prefix
# so the same .env works locally and inside Docker without manual overrides.
_logfire_base_url = settings.LOGFIRE_BASE_URL
if not _logfire_base_url and settings.LOGFIRE_TOKEN:
    if settings.LOGFIRE_TOKEN.startswith("pylf_v2_eu_"):
        _logfire_base_url = "https://logfire-eu.pydantic.dev"

logfire.configure(
    token=settings.LOGFIRE_TOKEN,
    advanced=logfire.AdvancedOptions(base_url=_logfire_base_url) if _logfire_base_url else None,
)

# Now safe to import app modules - logfire is already active
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, field_validator

from app.agents.graph import build_graph
from app.guardrails import guard, initialize_rails
from app.health import router as health_router
from app.logging import set_request_id
from app.services.cache import (
    get_exact_cached_response,
    get_semantic_cached_response,
    set_cached_response,
)
from app.services.health.connection_checker import check_all_connections, log_connection_summary
from app.services.retrieval.embedding import embed_query

# Custom Prometheus metrics
RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total number of /query requests",
    ["status"],
)
RAG_REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "Latency of /query requests in seconds",
)
GUARDRAILS_BLOCKS_TOTAL = Counter(
    "guardrails_blocks_total",
    "Number of requests blocked or allowed by guardrails",
    ["blocked"],
)

_security = HTTPBearer(auto_error=False)


def _init_rate_limiter():
    """Initialize rate limiting. Use Redis in production; fall back to in-memory storage locally."""
    from limits.storage import RedisStorage
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.extension import _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address

    try:
        storage = RedisStorage(settings.redis_url)
        # `storage.check()` returns False silently on some failures; ping the
        # underlying Redis client so we only use Redis when it is really reachable.
        if not storage.check() or not storage.storage.ping():
            raise ConnectionError("Redis did not respond to ping")
        app.state.limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)
        app.state.rate_limiter_storage = "redis"
        logfire.info("🚦 Rate limiting initialized via Redis.")
    except Exception as e:
        app.state.limiter = Limiter(key_func=get_remote_address)
        app.state.rate_limiter_storage = "memory"
        logfire.warning(f"⚠️ Redis unavailable ({e}); using in-memory rate limiting.")

    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return True


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(_security)):
    """
    Require a valid bearer token when RAG_API_KEY is configured.
    In development, omit RAG_API_KEY to disable authentication.
    """
    if not settings.API_KEY:
        # Development mode: no API key required.
        return None

    if not credentials or credentials.credentials != settings.API_KEY:
        logfire.warning("🔒 Unauthorized /query request: invalid or missing API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def _get_limiter_rule(times: int, seconds: int) -> str:
    """Convert times/seconds into a slowapi limit string, e.g. '20/minute'."""
    if seconds % 60 == 0:
        return f"{times}/{seconds // 60}minute"
    if seconds % 3600 == 0:
        return f"{times}/{seconds // 3600}hour"
    return f"{times}/{seconds}second"


class _AppLimiter:
    """
    Thin wrapper around the Limiter instance that is initialized at startup.
    Allows routes to be decorated at import time while the real limiter
    (Redis-backed or in-memory) is configured in startup_event.
    """

    def limit(self, rule_or_callable):
        def decorator(func):
            import functools

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                limiter = getattr(app.state, "limiter", None)
                if limiter is None:
                    return func(*args, **kwargs)

                rule = rule_or_callable() if callable(rule_or_callable) else rule_or_callable
                # Build the slowapi wrapper at request time so the limiter
                # instance and storage backend are always current.
                return limiter.limit(rule)(func)(*args, **kwargs)

            return wrapper

        return decorator


app_limiter = _AppLimiter()


def rate_limit(times: int = None, seconds: int = None):
    """
    Decorator factory that applies slowapi rate limiting using the limiter
    initialized at startup. Falls back to a no-op if the limiter is missing.
    The rule is resolved at request time so settings can be overridden in tests.
    """

    def _resolve_rule() -> str:
        t = times or settings.RATE_LIMIT_PER_MINUTE
        s = seconds or 60
        return _get_limiter_rule(t, s)

    return app_limiter.limit(_resolve_rule)


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")
app.include_router(health_router)

# Expose Prometheus metrics at /metrics with default request instrumentation.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.on_event("startup")
def startup_event():
    initialize_rails()

    # Build the agent graph with the production checkpointer (Postgres by default).
    app.state.rag_agent = build_graph()

    app.state.rate_limiter_enabled = _init_rate_limiter()

    # Verify all external dependencies are reachable.
    connection_results = check_all_connections()
    all_healthy = log_connection_summary(connection_results)
    if settings.STRICT_STARTUP and not all_healthy:
        failed = [name for name, r in connection_results.items() if not r.healthy]
        raise RuntimeError(f"STRICT_STARTUP enabled; failing services: {', '.join(failed)}")

    if not settings.API_KEY:
        logfire.warning("🔓 RAG_API_KEY is not set — /query is open to anyone. Set it in production.")


class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=20000)
    thread_id: str = Field(default="default_user", min_length=1)

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("thread_id must not be blank")
        if len(normalized) > settings.MEMORY_THREAD_ID_MAX_LENGTH:
            raise ValueError("thread_id exceeds the configured maximum length")
        return normalized


def _record_cached_turn(thread_id: str, question: str, response: dict[str, Any]) -> None:
    """Persist cache-hit turns so continuing a thread keeps its memory complete."""
    answer = response.get("answer")
    if not isinstance(answer, str):
        return

    try:
        app.state.rag_agent.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]},
        )
    except Exception as exc:
        logfire.warning(f"Could not persist cached conversation turn: {exc}")


def _serialize_messages(messages: list[Any]) -> list[dict[str, str]]:
    serialized = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "assistant")
            content = message.get("content", "")
        else:
            role = getattr(message, "type", "assistant")
            role = "user" if role == "human" else "assistant"
            content = getattr(message, "content", "")
        serialized.append({"role": str(role), "content": str(content)})
    return serialized


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image(_api_key: str = Depends(verify_api_key)):
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = app.state.rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


@app.get("/memory/{thread_id}")
def get_thread_memory(
    thread_id: str,
    _api_key: str = Depends(verify_api_key),
):
    """Return the retained conversation history for a LangGraph thread."""
    normalized_thread_id = thread_id.strip()
    if not normalized_thread_id or len(normalized_thread_id) > settings.MEMORY_THREAD_ID_MAX_LENGTH:
        raise HTTPException(status_code=422, detail="Invalid thread_id")

    try:
        snapshot = app.state.rag_agent.get_state({"configurable": {"thread_id": normalized_thread_id}})
        values = getattr(snapshot, "values", {}) or {}
        return {"thread_id": normalized_thread_id, "messages": _serialize_messages(values.get("messages", []))}
    except Exception as exc:
        logfire.error(f"Could not load thread memory: {exc}")
        raise HTTPException(status_code=503, detail="Conversation memory is unavailable") from exc


@app.get("/memory")
def list_thread_memories(
    _api_key: str = Depends(verify_api_key),
):
    """Return recent persisted conversations for the UI sidebar."""
    try:
        checkpointer = getattr(app.state.rag_agent, "checkpointer", None)
        if checkpointer is None:
            return {"threads": []}

        threads: dict[str, dict[str, Any]] = {}
        for item in checkpointer.list(None, limit=settings.MEMORY_THREAD_LIST_LIMIT * 4):
            configurable = item.config.get("configurable", {})
            thread_id = configurable.get("thread_id")
            if not thread_id or thread_id in threads:
                continue

            values = item.checkpoint.get("channel_values", {})
            messages = _serialize_messages(values.get("messages", []))
            user_messages = [message["content"] for message in messages if message["role"] == "user"]
            title = user_messages[0].strip() if user_messages else "New conversation"
            threads[thread_id] = {
                "thread_id": thread_id,
                "title": title[:80],
                "message_count": len(messages),
                "updated_at": item.checkpoint.get("ts"),
            }

            if len(threads) >= settings.MEMORY_THREAD_LIST_LIMIT:
                break

        return {"threads": list(threads.values())}
    except Exception as exc:
        logfire.error(f"Could not list conversation memory: {exc}")
        raise HTTPException(status_code=503, detail="Conversation list is unavailable") from exc


@app.post("/query")
@rate_limit()
def query(
    request: Request,
    body: QueryRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Runs the LangGraph RAG pipeline synchronously.
    Returns the final answer, thought process, status, and sources.
    """
    q = body.q
    thread_id = body.thread_id
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    start = time.perf_counter()
    with logfire.span("🔍 /query", request_id=request_id, thread_id=thread_id):
        # Gate: run guardrails synchronously so blocked requests never run the graph.
        rail_fired, rail_response = guard(q)
        if rail_fired:
            GUARDRAILS_BLOCKS_TOTAL.labels(blocked="true").inc()
            RAG_REQUESTS_TOTAL.labels(status="blocked").inc()
            RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
            logfire.info("🛡️ Request blocked by guardrails", request_id=request_id, thread_id=thread_id)
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
            }

        GUARDRAILS_BLOCKS_TOTAL.labels(blocked="false").inc()

        try:
            cached_response = get_exact_cached_response(q, thread_id)
            if cached_response is not None:
                _record_cached_turn(thread_id, q, cached_response)
                RAG_REQUESTS_TOTAL.labels(status="cache_hit").inc()
                RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
                logfire.info("✅ RAG response cache hit", request_id=request_id, thread_id=thread_id)
                return cached_response

            query_embedding = None
            try:
                query_embedding = embed_query(q)
                cached_response = get_semantic_cached_response(q, thread_id, query_embedding)
                if cached_response is not None:
                    _record_cached_turn(thread_id, q, cached_response)
                    RAG_REQUESTS_TOTAL.labels(status="cache_hit").inc()
                    RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
                    logfire.info("✅ RAG semantic response cache hit", request_id=request_id, thread_id=thread_id)
                    return cached_response
            except Exception as exc:
                logfire.warning(f"Query embedding unavailable; continuing without semantic cache: {exc}")

            rag_agent = app.state.rag_agent
            initial_state = {
                "messages": [{"role": "user", "content": q}],
                "current_query": q,
                "documents": [],
                "plan": ["Start"],
                "status": "Initializing Graph...",
                "query_embedding": query_embedding,
            }
            config = {"configurable": {"thread_id": thread_id}}
            final_output = rag_agent.invoke(initial_state, config=config)

            RAG_REQUESTS_TOTAL.labels(status="success").inc()
            RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
            logfire.info(
                "✅ RAG pipeline completed",
                request_id=request_id,
                thread_id=thread_id,
            )
            response = {
                "question": q,
                "answer": final_output.get("final_answer"),
                "thought_process": final_output.get("plan"),
                "status": final_output.get("status"),
                "sources": final_output.get("documents", []),
            }
            set_cached_response(q, thread_id, response, settings.CACHE_TTL_SECONDS, query_embedding)
            return response
        except Exception as e:
            RAG_REQUESTS_TOTAL.labels(status="error").inc()
            RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
            logfire.error(
                f"❌ RAG pipeline failed: {e}",
                request_id=request_id,
                thread_id=thread_id,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "request_id": request_id,
                    "status": "error",
                    "message": "Failed to process request. Please try again later.",
                },
            )
