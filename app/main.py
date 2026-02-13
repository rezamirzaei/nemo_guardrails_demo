"""
NeMo Guardrails FastAPI Application

A real-world implementation of NeMo Guardrails with:
- Input/Output rails for content moderation
- Topic control to keep conversations on-track
- Fact-checking capabilities
- Custom actions for business logic
- Web UI for interactive testing
- REST API with API key authentication
- Google Gemini primary backend with local Llama fallback
- MCP (Model Context Protocol) server support
"""

import asyncio
import os
import logging
import re
import secrets
from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from nemoguardrails import RailsConfig, LLMRails

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "config"
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
API_KEY_FILE = PROJECT_DIR / ".app_api_key"


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _get_non_placeholder_env(name: str, placeholder_values: set[str] | None = None) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        return ""
    normalized = value.lower()
    if placeholder_values and normalized in {p.lower() for p in placeholder_values}:
        return ""
    return value


GOOGLE_API_KEY = _get_non_placeholder_env("GOOGLE_API_KEY", {"your-gemini-api-key-here"})
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-lite").strip() or "gemini-2.0-flash-lite"
LOCAL_LLM_FALLBACK_ENABLED = _get_bool_env("LOCAL_LLM_FALLBACK_ENABLED", True)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.1:8b").strip() or "llama3.1:8b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
try:
    LOCAL_LLM_TIMEOUT_SECONDS = float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "90").strip())
except ValueError:
    LOCAL_LLM_TIMEOUT_SECONDS = 90.0
try:
    PRIMARY_LLM_TIMEOUT_SECONDS = float(os.getenv("PRIMARY_LLM_TIMEOUT_SECONDS", "20").strip())
except ValueError:
    PRIMARY_LLM_TIMEOUT_SECONDS = 20.0

LLM_BACKEND_GOOGLE = "google_guardrails"
LLM_BACKEND_LOCAL = "local_ollama"
LLM_BACKEND_UNAVAILABLE = "unavailable"

INPUT_REFUSAL_MESSAGE = (
    "I can't help with requests that try to bypass safety controls. "
    "Please rephrase your request."
)
OUTPUT_REFUSAL_MESSAGE = (
    "I can't provide that response because it violates safety checks. "
    "Please ask for a safer alternative."
)


def _load_persistent_api_key() -> str:
    if not API_KEY_FILE.exists():
        return ""
    try:
        return API_KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning(f"Failed to read API key file {API_KEY_FILE}: {exc}")
        return ""


def _store_persistent_api_key(key: str) -> bool:
    try:
        API_KEY_FILE.write_text(f"{key}\n", encoding="utf-8")
        os.chmod(API_KEY_FILE, 0o600)
        return True
    except OSError as exc:
        logger.warning(f"Failed to persist API key to {API_KEY_FILE}: {exc}")
        return False

# API key configuration.
# Default: require an API key.
API_KEY_ENABLED = _get_bool_env("API_KEY_REQUIRED", True)
_raw_api_key = os.getenv("APP_API_KEY", "").strip()
GENERATED_API_KEY = False
API_KEY_SOURCE = "disabled"
if API_KEY_ENABLED:
    if _raw_api_key and _raw_api_key != "your-app-api-key-here":
        API_KEY = _raw_api_key
        API_KEY_SOURCE = "env"
    else:
        persisted_key = _load_persistent_api_key()
        if persisted_key:
            API_KEY = persisted_key
            API_KEY_SOURCE = "file"
        else:
            API_KEY = secrets.token_urlsafe(32)
            GENERATED_API_KEY = True
            if _store_persistent_api_key(API_KEY):
                API_KEY_SOURCE = "generated_file"
            else:
                API_KEY_SOURCE = "generated_ephemeral"
else:
    API_KEY = None
UI_AUTH_COOKIE_ENABLED = _get_bool_env("UI_AUTH_COOKIE_ENABLED", True)
UI_AUTH_COOKIE_NAME = os.getenv("UI_AUTH_COOKIE_NAME", "guardrails_ui_api_key").strip() or "guardrails_ui_api_key"
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global rails instance
rails: Optional[LLMRails] = None
llm_backend_mode = LLM_BACKEND_UNAVAILABLE
llm_backend_model = ""

# In-memory conversation history (best-effort, for UI/demo use).
_CONVERSATION_LOCK = asyncio.Lock()
_CONVERSATIONS: dict[str, dict[str, object]] = {}
_MAX_CONVERSATIONS = 200
_MAX_MESSAGES_PER_CONVERSATION = 40
_CONVERSATION_TTL_SECONDS = 6 * 60 * 60  # 6 hours


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _new_conversation_id() -> str:
    return uuid4().hex


def _ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _cleanup_conversations(now: float) -> None:
    expired = [
        cid
        for cid, entry in _CONVERSATIONS.items()
        if isinstance(entry.get("updated_at"), (int, float))
        and now - float(entry["updated_at"]) > _CONVERSATION_TTL_SECONDS
    ]
    for cid in expired:
        _CONVERSATIONS.pop(cid, None)


async def _peek_conversation(conversation_id: str) -> tuple[list[dict[str, str]], float | None]:
    async with _CONVERSATION_LOCK:
        now = _now_ts()
        _cleanup_conversations(now)

        entry = _CONVERSATIONS.get(conversation_id)
        if entry is None:
            return [], None

        updated_at = entry.get("updated_at")
        ts = float(updated_at) if isinstance(updated_at, (int, float)) else None

        messages = entry.get("messages")
        if isinstance(messages, deque):
            return list(messages), ts
        return [], ts


async def _delete_conversation(conversation_id: str) -> bool:
    async with _CONVERSATION_LOCK:
        now = _now_ts()
        _cleanup_conversations(now)
        return _CONVERSATIONS.pop(conversation_id, None) is not None


async def _get_conversation_messages(conversation_id: str) -> list[dict[str, str]]:
    async with _CONVERSATION_LOCK:
        now = _now_ts()
        _cleanup_conversations(now)

        entry = _CONVERSATIONS.get(conversation_id)
        if entry is None:
            entry = {
                "messages": deque(maxlen=_MAX_MESSAGES_PER_CONVERSATION),
                "updated_at": now,
            }
            _CONVERSATIONS[conversation_id] = entry

        entry["updated_at"] = now
        messages = entry.get("messages")
        if isinstance(messages, deque):
            return list(messages)
        return []


async def _set_conversation_messages(conversation_id: str, messages: list[dict[str, str]]) -> None:
    async with _CONVERSATION_LOCK:
        now = _now_ts()
        _cleanup_conversations(now)

        # Evict oldest if we exceed cap
        if len(_CONVERSATIONS) >= _MAX_CONVERSATIONS and conversation_id not in _CONVERSATIONS:
            oldest_id = None
            oldest_ts = None
            for cid, entry in _CONVERSATIONS.items():
                ts = entry.get("updated_at")
                if not isinstance(ts, (int, float)):
                    continue
                if oldest_ts is None or float(ts) < float(oldest_ts):
                    oldest_ts = float(ts)
                    oldest_id = cid
            if oldest_id is not None:
                _CONVERSATIONS.pop(oldest_id, None)

        entry = _CONVERSATIONS.get(conversation_id)
        if entry is None or not isinstance(entry.get("messages"), deque):
            entry = {
                "messages": deque(maxlen=_MAX_MESSAGES_PER_CONVERSATION),
                "updated_at": now,
            }
            _CONVERSATIONS[conversation_id] = entry

        dq = entry["messages"]
        if isinstance(dq, deque):
            dq.clear()
            for m in messages[-_MAX_MESSAGES_PER_CONVERSATION :]:
                role = m.get("role")
                content = m.get("content")
                if role in {"user", "assistant"} and isinstance(content, str):
                    dq.append({"role": role, "content": content})
        entry["updated_at"] = now


def _is_google_backend_configured() -> bool:
    return bool(GOOGLE_API_KEY)


def _backend_unavailable_detail() -> str:
    return (
        "No LLM backend is available. Configure GOOGLE_API_KEY for Gemini, "
        "or enable LOCAL_LLM_FALLBACK_ENABLED and run Ollama locally."
    )


async def _check_local_llm_reachable() -> bool:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        return response.status_code == 200
    except Exception:
        return False


async def _list_ollama_models(client: httpx.AsyncClient | None = None) -> list[str]:
    async def _fetch(async_client: httpx.AsyncClient) -> list[str]:
        response = await async_client.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models")
        if not isinstance(models, list):
            return []
        names: list[str] = []
        for model in models:
            if not isinstance(model, dict):
                continue
            name = model.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return names

    try:
        if client is not None:
            return await _fetch(client)
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as async_client:
            return await _fetch(async_client)
    except Exception:
        return []


def _select_best_local_model(available_models: list[str]) -> str | None:
    if not available_models:
        return None

    preferred = [
        LOCAL_LLM_MODEL,
        "llama3.2:3b",
        "llama3.1:8b",
        "llama3.2:1b",
        "phi3:mini",
    ]
    available_lower = {m.lower(): m for m in available_models}
    for model in preferred:
        selected = available_lower.get(model.lower())
        if selected:
            return selected
    return available_models[0]


async def _generate_with_ollama(messages: list[dict[str, str]]) -> str:
    model_to_use = LOCAL_LLM_MODEL
    payload: dict[str, Any] = {
        "model": model_to_use,
        "messages": messages,
        "stream": False,
    }
    timeout = httpx.Timeout(
        timeout=LOCAL_LLM_TIMEOUT_SECONDS,
        connect=min(10.0, LOCAL_LLM_TIMEOUT_SECONDS),
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        if response.status_code == 404:
            try:
                err = str(response.json().get("error", "")).lower()
            except Exception:
                err = response.text.lower()
            if "model" in err and "not found" in err:
                available_models = await _list_ollama_models(client)
                fallback_model = _select_best_local_model(available_models)
                if fallback_model and fallback_model.lower() != model_to_use.lower():
                    logger.warning(
                        f"Ollama model '{model_to_use}' not found. "
                        f"Retrying with '{fallback_model}'."
                    )
                    model_to_use = fallback_model
                    payload["model"] = model_to_use
                    response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                else:
                    available_text = ", ".join(available_models) if available_models else "(none)"
                    raise ValueError(
                        f"Ollama model '{LOCAL_LLM_MODEL}' is not available. "
                        f"Available models: {available_text}."
                    )
        response.raise_for_status()

    body = response.json()
    message = body.get("message")
    content = message.get("content") if isinstance(message, dict) else body.get("response")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Local LLM returned an empty response")

    return content.strip()


async def _generate_with_runtime_failover(messages: list[dict[str, str]]) -> tuple[str, str]:
    """
    Generate text using the active backend, with per-request failover to local Ollama.

    This allows automatic fallback when Gemini is unavailable, misconfigured, rate limited,
    or returns an empty response.
    """
    if llm_backend_mode == LLM_BACKEND_GOOGLE and rails is not None:
        try:
            response = await asyncio.wait_for(
                rails.generate_async(messages=messages),
                timeout=PRIMARY_LLM_TIMEOUT_SECONDS,
            )
            response_text = str(response.get("content", "")).strip()
            if response_text:
                return response_text, LLM_BACKEND_GOOGLE

            logger.warning(
                "Gemini/Guardrails backend returned empty content; attempting local Ollama fallback."
            )
        except Exception as exc:
            if not LOCAL_LLM_FALLBACK_ENABLED:
                raise
            logger.warning(
                f"Gemini/Guardrails request failed ({type(exc).__name__}: {exc}); "
                "attempting local Ollama fallback."
            )

    if llm_backend_mode == LLM_BACKEND_GOOGLE and rails is None:
        logger.warning(
            "Gemini backend mode is active but rails are not initialized; "
            "attempting local Ollama fallback."
        )

    if llm_backend_mode == LLM_BACKEND_LOCAL:
        response_text = await _generate_with_ollama(messages)
        return response_text, LLM_BACKEND_LOCAL

    if LOCAL_LLM_FALLBACK_ENABLED:
        response_text = await _generate_with_ollama(messages)
        return response_text, LLM_BACKEND_LOCAL

    raise HTTPException(status_code=503, detail=_backend_unavailable_detail())


def _action_result_to_bool(result: Any) -> Optional[bool]:
    if isinstance(result, bool):
        return result

    return_value = getattr(result, "return_value", None)
    if isinstance(return_value, bool):
        return return_value

    return None


async def _run_guardrails_action_bool(action_name: str, context: dict[str, str]) -> Optional[bool]:
    if rails is None:
        return None

    runtime = getattr(rails, "runtime", None)
    if runtime is None:
        return None

    action_dispatcher = getattr(runtime, "action_dispatcher", None)
    llm_task_manager = getattr(runtime, "llm_task_manager", None)
    if action_dispatcher is None or llm_task_manager is None:
        return None

    llm = getattr(rails, "llm", None)
    if llm is None:
        return None

    if not action_dispatcher.has_registered(action_name):
        return None

    params: dict[str, Any] = {
        "context": context,
        "llm_task_manager": llm_task_manager,
        "llm": llm,
        "config": rails.config,
    }

    result, status = await action_dispatcher.execute_action(action_name, params)
    if status != "success":
        return None

    return _action_result_to_bool(result)


async def _run_input_safety(message: str) -> tuple[bool, bool, list[str]]:
    from config.actions import check_jailbreak_attempt, keyword_input_filter

    context = {"user_message": message}
    is_jailbreak = bool(await check_jailbreak_attempt(context))
    keyword_allowed = bool(await keyword_input_filter(context))

    semantic_allowed = True
    semantic_result = await _run_guardrails_action_bool("self_check_input", context)
    if semantic_result is not None:
        semantic_allowed = bool(semantic_result)

    input_allowed = keyword_allowed and semantic_allowed

    details: list[str] = []
    if is_jailbreak:
        details.append("Jailbreak attempt detected")
    if not keyword_allowed:
        details.append("Input blocked by keyword prefilter")
    if not semantic_allowed:
        details.append("Input blocked by NeMo self-check")
    if not details:
        details.append("Message passes basic input checks")

    return (input_allowed and not is_jailbreak), is_jailbreak, details


async def _run_output_safety(message: str) -> tuple[bool, list[str]]:
    from config.actions import check_facts, keyword_output_filter

    keyword_allowed = bool(await keyword_output_filter({"bot_message": message}))
    semantic_allowed = True
    semantic_result = await _run_guardrails_action_bool("self_check_output", {"bot_message": message})
    if semantic_result is not None:
        semantic_allowed = bool(semantic_result)

    output_allowed = keyword_allowed and semantic_allowed
    facts_ok = bool(await check_facts({"bot_message": message}))

    details: list[str] = []
    if not keyword_allowed:
        details.append("Output blocked by keyword prefilter")
    if not semantic_allowed:
        details.append("Output blocked by NeMo self-check")
    if not facts_ok:
        details.append("Output contains unverifiable claims")
    if not details:
        details.append("Message passes basic output checks")

    return output_allowed and facts_ok, details


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=4096, description="User message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Hello, how can you help me today?",
                "conversation_id": "conv-123"
            }
        }
    }


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Assistant response")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    guardrails_triggered: bool = Field(False, description="Whether any guardrails were triggered")
    backend_used: str = Field(LLM_BACKEND_UNAVAILABLE, description="Backend used for this response")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "Hello! I'm here to help you with various tasks...",
                "conversation_id": "conv-123",
                "guardrails_triggered": False,
                "backend_used": "google_guardrails",
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    guardrails_loaded: bool
    llm_backend_mode: str = LLM_BACKEND_UNAVAILABLE
    llm_backend_model: str = ""
    local_fallback_enabled: bool = LOCAL_LLM_FALLBACK_ENABLED
    version: str = "1.0.0"


class APIKeyResponse(BaseModel):
    """Response model for API key info"""
    message: str
    api_key_required: bool = True
    ui_cookie_auth_enabled: bool = False
    header_name: str = "X-API-Key"
    llm_backend_mode: str = LLM_BACKEND_UNAVAILABLE
    llm_backend_model: str = ""


class SafetyCheckRequest(BaseModel):
    """Request model for safety check endpoints."""

    message: str = Field(..., min_length=1, max_length=4096, description="Message to check")


class InputSafetyResponse(BaseModel):
    """Response model for input safety checks."""

    message: str
    is_safe: bool
    jailbreak_detected: bool
    input_allowed: bool
    details: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OutputSafetyResponse(BaseModel):
    """Response model for output safety checks."""

    message: str
    is_safe: bool
    output_allowed: bool
    details: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationMessage(BaseModel):
    """Conversation message model."""

    role: str
    content: str


class ConversationResponse(BaseModel):
    """Conversation history response."""

    conversation_id: str
    messages: list[ConversationMessage]
    updated_at: str


class ConversationCreateResponse(BaseModel):
    """Response for creating a new conversation."""

    conversation_id: str


class ConversationDeleteResponse(BaseModel):
    """Response for deleting a conversation."""

    conversation_id: str
    deleted: bool


# =============================================================================
# Authentication
# =============================================================================

async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(API_KEY_HEADER),
) -> Optional[str]:
    """Verify API key for protected endpoints. If auth is disabled, allow all requests."""
    if not API_KEY_ENABLED:
        return None

    header_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
    cookie_key = None
    if UI_AUTH_COOKIE_ENABLED:
        cookie_value = request.cookies.get(UI_AUTH_COOKIE_NAME, "").strip()
        if cookie_value:
            cookie_key = cookie_value

    if header_key and secrets.compare_digest(header_key, API_KEY):
        return header_key
    if cookie_key and secrets.compare_digest(cookie_key, API_KEY):
        return cookie_key

    if header_key or cookie_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    raise HTTPException(
        status_code=401,
        detail="API key is required. Pass it in the X-API-Key header.",
        headers={"WWW-Authenticate": "ApiKey"}
    )


# =============================================================================
# Application Lifespan
# =============================================================================

def load_rails() -> LLMRails:
    """Load NeMo Guardrails config from disk, with environment overrides."""
    config = RailsConfig.from_path(str(CONFIG_DIR))

    for model in config.models:
        if getattr(model, "type", None) == "main":
            model.model = GOOGLE_MODEL

    return LLMRails(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - loads guardrails on startup"""
    global rails, llm_backend_mode, llm_backend_model

    logger.info("Loading NeMo Guardrails configuration...")
    if API_KEY_ENABLED:
        if API_KEY_SOURCE == "env":
            logger.info("API Key authentication: ENABLED (using APP_API_KEY)")
        elif API_KEY_SOURCE == "file":
            logger.info(f"API Key authentication: ENABLED (using persisted key from {API_KEY_FILE})")
        elif API_KEY_SOURCE == "generated_file":
            logger.info(f"API Key authentication: ENABLED (generated key: {API_KEY})")
            logger.info(f"Key persisted at {API_KEY_FILE} for stable restarts.")
        elif API_KEY_SOURCE == "generated_ephemeral":
            logger.info(f"API Key authentication: ENABLED (generated key: {API_KEY})")
            logger.info("Set APP_API_KEY in .env to keep a stable key across restarts/reloads.")
        else:
            logger.info("API Key authentication: ENABLED")
    else:
        logger.info("API Key authentication: DISABLED (no API key required)")

    rails = None
    llm_backend_mode = LLM_BACKEND_UNAVAILABLE
    llm_backend_model = ""

    if _is_google_backend_configured():
        try:
            rails = load_rails()
            llm_backend_mode = LLM_BACKEND_GOOGLE
            llm_backend_model = GOOGLE_MODEL
            logger.info(f"NeMo Guardrails loaded successfully with {GOOGLE_MODEL}.")
        except Exception as exc:
            if LOCAL_LLM_FALLBACK_ENABLED:
                llm_backend_mode = LLM_BACKEND_LOCAL
                llm_backend_model = LOCAL_LLM_MODEL
                logger.warning(f"Failed to initialize Gemini/Guardrails backend: {exc}")
                logger.warning(
                    f"Falling back to local Ollama backend at {OLLAMA_BASE_URL} "
                    f"(model={LOCAL_LLM_MODEL})."
                )
            else:
                logger.error(f"Failed to initialize Gemini/Guardrails backend: {exc}")
                raise
    elif LOCAL_LLM_FALLBACK_ENABLED:
        llm_backend_mode = LLM_BACKEND_LOCAL
        llm_backend_model = LOCAL_LLM_MODEL
        logger.info(
            f"GOOGLE_API_KEY is not set; using local Ollama backend at "
            f"{OLLAMA_BASE_URL} (model={LOCAL_LLM_MODEL})."
        )
    else:
        logger.warning(_backend_unavailable_detail())

    if llm_backend_mode == LLM_BACKEND_LOCAL:
        if await _check_local_llm_reachable():
            logger.info("Local Ollama backend is reachable.")
        else:
            logger.warning(
                f"Ollama is not reachable at {OLLAMA_BASE_URL}. "
                "Chat requests may return 503 until it is available."
            )

    yield

    # Cleanup on shutdown
    logger.info("Shutting down NeMo Guardrails application...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="NeMo Guardrails API",
    description="""
## NeMo Guardrails REST API

A production-ready NeMo Guardrails application with input/output moderation.

### Features
- 🛡️ **Input Rails**: Content moderation, jailbreak detection
- 🔒 **Output Rails**: Response validation, fact-checking
- 🔑 **API Key Authentication**: Secure access to chat endpoints
- 💬 **Web UI**: Interactive chat interface for testing
- 🤖 **Dual Backend**: Gemini + Guardrails when configured, local Ollama fallback otherwise
- 🔌 **MCP Support**: Model Context Protocol server for Claude Desktop integration

### Authentication
All `/api/*` endpoints require an API key passed in the `X-API-Key` header by default.
Set `API_KEY_REQUIRED=false` to disable API key auth.
Check the server logs for the generated API key, or set `APP_API_KEY` in your `.env` file.

### MCP Integration
This app also runs as an MCP server. See `/mcp/info` for configuration details.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
def parse_cors_origins(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


CORS_ORIGINS = parse_cors_origins(os.getenv("CORS_ORIGINS", "*"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    # API key auth uses headers, not cookies. Keeping this off avoids invalid "*" + credentials CORS.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response compression (helps the AngularJS UI and JSON responses).
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Lightweight security headers (kept permissive; tighten CSP separately if needed).
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    return response

# Serve the AngularJS UI and assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# Public Endpoints (No Auth Required)
# =============================================================================

@app.get("/", include_in_schema=False)
async def root(request: Request):
    """Serve the main web UI"""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="UI is missing: app/static/index.html")
    response = FileResponse(index_path)
    if API_KEY_ENABLED and UI_AUTH_COOKIE_ENABLED and API_KEY:
        response.set_cookie(
            key=UI_AUTH_COOKIE_NAME,
            value=API_KEY,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
            max_age=7 * 24 * 60 * 60,
            path="/",
        )
    return response


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint (no authentication required)

    Returns the health status and whether guardrails are loaded.
    """
    return HealthResponse(
        status="healthy" if llm_backend_mode != LLM_BACKEND_UNAVAILABLE else "degraded",
        guardrails_loaded=rails is not None,
        llm_backend_mode=llm_backend_mode,
        llm_backend_model=llm_backend_model,
        local_fallback_enabled=LOCAL_LLM_FALLBACK_ENABLED,
    )


@app.get("/api/info", response_model=APIKeyResponse, tags=["Info"])
async def api_info():
    """
    Get API information (no authentication required)

    Returns information about how to authenticate with the API.
    """
    return APIKeyResponse(
        message=(
            (
                f"Use the X-API-Key header. Local key is persisted in {API_KEY_FILE} "
                f"unless APP_API_KEY is set. Browser UI auto-auth is "
                f"{'enabled' if UI_AUTH_COOKIE_ENABLED else 'disabled'}."
            )
            if API_KEY_ENABLED
            else "API key authentication is disabled."
        ),
        api_key_required=API_KEY_ENABLED,
        ui_cookie_auth_enabled=(API_KEY_ENABLED and UI_AUTH_COOKIE_ENABLED),
        header_name="X-API-Key",
        llm_backend_mode=llm_backend_mode,
        llm_backend_model=llm_backend_model,
    )


@app.get("/mcp/info", tags=["MCP"])
async def mcp_info():
    """
    Get MCP (Model Context Protocol) configuration information.

    Returns the configuration needed to add this server to Claude Desktop or other MCP clients.
    """
    repo_dir = str(PROJECT_DIR)
    return {
        "name": "nemo-guardrails-mcp",
        "description": (
            "NeMo Guardrails MCP Server - Gemini+Guardrails primary backend "
            "with local Ollama fallback"
        ),
        "active_backend": {
            "mode": llm_backend_mode,
            "model": llm_backend_model,
            "local_fallback_enabled": LOCAL_LLM_FALLBACK_ENABLED,
        },
        "tools": [
            {
                "name": "chat_with_guardrails",
                "description": "Send a message to the AI assistant with guardrails protection"
            },
            {
                "name": "check_input_safety",
                "description": "Check if a message would pass the input guardrails"
            },
            {
                "name": "get_guardrails_config",
                "description": "Get information about the current guardrails configuration"
            }
        ],
        "claude_desktop_config": {
            "mcpServers": {
                "nemo-guardrails": {
                    "command": "uv",
                    "args": [
                        "run",
                        "--directory",
                        repo_dir,
                        "python",
                        "-m",
                        "app.mcp_server"
                    ],
                    "env": {
                        "GOOGLE_API_KEY": "your-gemini-api-key-here",
                        "LOCAL_LLM_FALLBACK_ENABLED": "true",
                        "LOCAL_LLM_MODEL": "llama3.1:8b",
                        "OLLAMA_BASE_URL": "http://127.0.0.1:11434"
                    }
                }
            }
        },
        "setup_instructions": [
            "1. Copy the 'claude_desktop_config' JSON above",
            "2. Open Claude Desktop settings",
            "3. Go to Developer > MCP Servers",
            "4. Paste the configuration and replace GOOGLE_API_KEY (or keep fallback-only mode)",
            "5. Restart Claude Desktop"
        ]
    }


# =============================================================================
# Protected API Endpoints (API Key Required)
# =============================================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Send a chat message",
    description="Send a message to the AI assistant with guardrails protection."
)
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Chat endpoint with NeMo Guardrails protection.

    This endpoint processes user messages through:
    1. **Input rails** - Content moderation, jailbreak detection
    2. **LLM generation** - AI response generation
    3. **Output rails** - Response validation, fact-checking

    Requires API key authentication via `X-API-Key` header.
    """
    conversation_id = (request.conversation_id or "").strip() or _new_conversation_id()
    logger.info(f"Processing message (conversation_id={conversation_id}): {request.message[:50]}...")

    try:
        history = await _get_conversation_messages(conversation_id)
        messages = history + [{"role": "user", "content": request.message}]

        guardrails_triggered = False
        backend_used = llm_backend_mode
        try:
            input_is_safe, _, input_details = await _run_input_safety(request.message)
        except Exception as exc:
            logger.warning(f"Input safety checks unavailable: {exc}")
            input_is_safe = True
            input_details = ["Input safety checks unavailable"]

        if not input_is_safe:
            response_text = INPUT_REFUSAL_MESSAGE
            guardrails_triggered = True
            logger.info(
                f"Input blocked by safety checks (conversation_id={conversation_id}, details={input_details})"
            )
            backend_used = "guardrails_block"
        else:
            response_text, backend_used = await _generate_with_runtime_failover(messages)

        # Run output checks for both primary and fallback generation paths.
        try:
            output_is_safe, output_details = await _run_output_safety(response_text)
        except Exception as exc:
            logger.warning(f"Output safety checks unavailable: {exc}")
            output_is_safe, output_details = True, ["Output safety checks unavailable"]

        if not output_is_safe:
            guardrails_triggered = True
            logger.info(
                f"Output blocked by safety checks (conversation_id={conversation_id}, details={output_details})"
            )
            response_text = OUTPUT_REFUSAL_MESSAGE

        refusal_hints = (
            "I cannot",
            "I'm not able to",
            "I'm sorry, but I can't help with that request",
            "I notice you might be trying to bypass my guidelines",
        )
        if any(hint in response_text for hint in refusal_hints):
            guardrails_triggered = True

        logger.info(
            f"Generated response (backend_used={backend_used}, "
            f"guardrails_triggered={guardrails_triggered})"
        )

        await _set_conversation_messages(
            conversation_id,
            messages + [{"role": "assistant", "content": response_text}],
        )

        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            guardrails_triggered=guardrails_triggered,
            backend_used=backend_used,
        )

    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e)
        logger.error(f"Error generating response: {e}")

        if isinstance(e, ValueError):
            lowered = error_str.lower()
            if "ollama model" in lowered or "local llm returned an empty response" in lowered:
                raise HTTPException(status_code=503, detail=error_str)

        if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Local Llama fallback is unavailable. Start Ollama and ensure "
                    f"model '{LOCAL_LLM_MODEL}' is pulled."
                ),
            )
        if isinstance(e, httpx.HTTPStatusError):
            raise HTTPException(
                status_code=502,
                detail=f"Local Llama backend returned HTTP {e.response.status_code}.",
            )

        # Handle rate limit errors gracefully
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            # Extract retry time if available
            retry_match = re.search(r"retry in (\d+)", error_str.lower())
            retry_time = retry_match.group(1) if retry_match else "60"

            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. The free tier has limited requests per day. Please wait {retry_time} seconds and try again, or upgrade your Google AI API plan."
            )

        raise HTTPException(status_code=500, detail=f"Error processing request: {error_str}")


@app.post(
    "/api/chat/test",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Test chat without LLM (mock response)",
    description="Test the API without making actual LLM calls. Useful for testing authentication and request format."
)
async def chat_test(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Test endpoint that returns a mock response without calling the LLM.
    Useful for testing authentication and the API interface.
    """
    conversation_id = (request.conversation_id or "").strip() or _new_conversation_id()
    history = await _get_conversation_messages(conversation_id)
    messages = history + [{"role": "user", "content": request.message}]

    response_text = (
        f"[TEST MODE] Received your message: '{request.message[:100]}...' "
        "- this endpoint does not call the LLM."
    )

    await _set_conversation_messages(
        conversation_id,
        messages + [{"role": "assistant", "content": response_text}],
    )

    return ChatResponse(
        response=response_text,
        conversation_id=conversation_id,
        guardrails_triggered=False,
        backend_used="mock",
    )


@app.post(
    "/api/tools/check_input_safety",
    response_model=InputSafetyResponse,
    tags=["Tools"],
    summary="Check input safety (no LLM call)",
)
async def check_input_safety(
    request: SafetyCheckRequest,
    api_key: str = Depends(verify_api_key),
):
    """Run lightweight input checks without calling the LLM."""
    is_safe, is_jailbreak, details = await _run_input_safety(request.message)
    input_allowed = not any(detail.startswith("Input blocked") for detail in details)

    return InputSafetyResponse(
        message=request.message,
        is_safe=is_safe,
        jailbreak_detected=is_jailbreak,
        input_allowed=input_allowed,
        details=details,
    )


@app.post(
    "/api/tools/check_output_safety",
    response_model=OutputSafetyResponse,
    tags=["Tools"],
    summary="Check output safety (no LLM call)",
)
async def check_output_safety(
    request: SafetyCheckRequest,
    api_key: str = Depends(verify_api_key),
):
    """Run lightweight output checks without calling the LLM."""
    output_allowed, details = await _run_output_safety(request.message)

    return OutputSafetyResponse(
        message=request.message,
        is_safe=output_allowed,
        output_allowed=output_allowed,
        details=details,
    )


@app.post(
    "/api/conversations",
    response_model=ConversationCreateResponse,
    tags=["Conversations"],
    summary="Create a new conversation",
)
async def create_conversation(api_key: str = Depends(verify_api_key)):
    conversation_id = _new_conversation_id()
    await _set_conversation_messages(conversation_id, [])
    return ConversationCreateResponse(conversation_id=conversation_id)


@app.get(
    "/api/conversations/{conversation_id}",
    response_model=ConversationResponse,
    tags=["Conversations"],
    summary="Get conversation history",
)
async def get_conversation(
    conversation_id: str,
    api_key: str = Depends(verify_api_key),
):
    messages, updated_at = await _peek_conversation(conversation_id)
    if updated_at is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        conversation_id=conversation_id,
        messages=[ConversationMessage(**m) for m in messages],
        updated_at=_ts_to_iso(updated_at),
    )


@app.delete(
    "/api/conversations/{conversation_id}",
    response_model=ConversationDeleteResponse,
    tags=["Conversations"],
    summary="Delete a conversation",
)
async def delete_conversation(
    conversation_id: str,
    api_key: str = Depends(verify_api_key),
):
    deleted = await _delete_conversation(conversation_id)
    return ConversationDeleteResponse(conversation_id=conversation_id, deleted=deleted)


@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Streaming chat endpoint (placeholder for streaming implementation)

    Note: Streaming with guardrails requires careful handling
    as output rails need the full response to validate.
    """
    raise HTTPException(
        status_code=501,
        detail="Streaming not yet implemented with full guardrails support"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )
