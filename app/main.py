"""
NeMo Guardrails FastAPI Application

A real-world implementation of NeMo Guardrails with:
- Input/Output rails for content moderation
- Topic control to keep conversations on-track
- Fact-checking capabilities
- Custom actions for business logic
- Web UI for interactive testing
- REST API with API key authentication
- Google Gemini as the LLM backend
- MCP (Model Context Protocol) server support
"""

import asyncio
import os
import logging
import secrets
from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends
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

# API key configuration.
# Default: require an API key, and generate a random one if not provided.
API_KEY_ENABLED = os.getenv("API_KEY_REQUIRED", "true").strip().lower() in {"1", "true", "yes", "on"}
_raw_api_key = os.getenv("APP_API_KEY", "").strip()
GENERATED_API_KEY = False
if API_KEY_ENABLED:
    if not _raw_api_key or _raw_api_key == "your-app-api-key-here":
        API_KEY = secrets.token_urlsafe(32)
        GENERATED_API_KEY = True
    else:
        API_KEY = _raw_api_key
else:
    API_KEY = None
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global rails instance
rails: Optional[LLMRails] = None

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
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "Hello! I'm here to help you with various tasks...",
                "conversation_id": "conv-123",
                "guardrails_triggered": False,
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    guardrails_loaded: bool
    version: str = "1.0.0"


class APIKeyResponse(BaseModel):
    """Response model for API key info"""
    message: str
    api_key_required: bool = True
    header_name: str = "X-API-Key"


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

async def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)) -> Optional[str]:
    """Verify API key for protected endpoints. If auth is disabled, allow all requests."""
    if not API_KEY_ENABLED:
        return None

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key is required. Pass it in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key


# =============================================================================
# Application Lifespan
# =============================================================================

def load_rails() -> LLMRails:
    """Load NeMo Guardrails config from disk, with environment overrides."""
    config = RailsConfig.from_path(str(CONFIG_DIR))

    model_override = os.getenv("GOOGLE_MODEL", "").strip()
    if model_override:
        for model in config.models:
            if getattr(model, "type", None) == "main":
                model.model = model_override

    return LLMRails(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - loads guardrails on startup"""
    global rails

    logger.info("Loading NeMo Guardrails configuration...")
    if API_KEY_ENABLED:
        if GENERATED_API_KEY:
            logger.info(f"API Key authentication: ENABLED (generated key: {API_KEY})")
            logger.info("Set APP_API_KEY in .env to keep a stable key across restarts/reloads.")
        else:
            logger.info("API Key authentication: ENABLED (using APP_API_KEY)")
    else:
        logger.info("API Key authentication: DISABLED (no API key required)")

    try:
        rails = load_rails()
        logger.info("NeMo Guardrails loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load guardrails: {e}")
        raise

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

A production-ready NeMo Guardrails application with input/output moderation, powered by **Google Gemini**.

### Features
- 🛡️ **Input Rails**: Content moderation, jailbreak detection
- 🔒 **Output Rails**: Response validation, fact-checking
- 🔑 **API Key Authentication**: Secure access to chat endpoints
- 💬 **Web UI**: Interactive chat interface for testing
- 🤖 **Google Gemini**: Powered by Gemini (configurable)
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
async def root():
    """Serve the main web UI"""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="UI is missing: app/static/index.html")
    return FileResponse(index_path)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint (no authentication required)

    Returns the health status and whether guardrails are loaded.
    """
    return HealthResponse(
        status="healthy",
        guardrails_loaded=rails is not None
    )


@app.get("/api/info", response_model=APIKeyResponse, tags=["Info"])
async def api_info():
    """
    Get API information (no authentication required)

    Returns information about how to authenticate with the API.
    """
    return APIKeyResponse(
        message=(
            "Use the X-API-Key header to authenticate. Check server logs for the API key."
            if API_KEY_ENABLED
            else "API key authentication is disabled."
        ),
        api_key_required=API_KEY_ENABLED,
        header_name="X-API-Key"
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
        "description": "NeMo Guardrails MCP Server - AI assistant with safety rails powered by Google Gemini",
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
                        "GOOGLE_API_KEY": "your-gemini-api-key-here"
                    }
                }
            }
        },
        "setup_instructions": [
            "1. Copy the 'claude_desktop_config' JSON above",
            "2. Open Claude Desktop settings",
            "3. Go to Developer > MCP Servers",
            "4. Paste the configuration and replace the GOOGLE_API_KEY with your actual key",
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
    if rails is None:
        raise HTTPException(status_code=503, detail="Guardrails not initialized")

    conversation_id = (request.conversation_id or "").strip() or _new_conversation_id()
    logger.info(f"Processing message (conversation_id={conversation_id}): {request.message[:50]}...")

    try:
        history = await _get_conversation_messages(conversation_id)
        messages = history + [{"role": "user", "content": request.message}]

        guardrails_triggered = False
        try:
            from config.actions import check_jailbreak_attempt, self_check_input

            context = {"user_message": request.message}
            is_jailbreak = await check_jailbreak_attempt(context)
            input_allowed = await self_check_input(context)
            guardrails_triggered = bool(is_jailbreak) or not bool(input_allowed)
        except Exception as e:
            logger.debug(f"Input safety checks unavailable: {e}")

        # Generate response through guardrails
        response = await rails.generate_async(
            messages=messages
        )

        response_text = response.get("content", "")

        # Check for guardrail intervention indicators
        refusal_hints = (
            "I cannot",
            "I'm not able to",
            "I'm sorry, but I can't help with that request",
            "I notice you might be trying to bypass my guidelines",
        )
        if any(hint in response_text for hint in refusal_hints):
            guardrails_triggered = True

        logger.info(f"Generated response (guardrails_triggered={guardrails_triggered})")

        await _set_conversation_messages(
            conversation_id,
            messages + [{"role": "assistant", "content": response_text}],
        )

        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            guardrails_triggered=guardrails_triggered
        )

    except Exception as e:
        error_str = str(e)
        logger.error(f"Error generating response: {e}")

        # Handle rate limit errors gracefully
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            # Extract retry time if available
            import re
            retry_match = re.search(r'retry in (\d+)', error_str.lower())
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
        guardrails_triggered=False
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
    from config.actions import check_jailbreak_attempt, self_check_input

    context = {"user_message": request.message}
    is_jailbreak = bool(await check_jailbreak_attempt(context))
    input_allowed = bool(await self_check_input(context))

    details: list[str] = []
    if is_jailbreak:
        details.append("Jailbreak attempt detected")
    if not input_allowed:
        details.append("Input contains blocked patterns")
    if not details:
        details.append("Message passes basic input checks")

    return InputSafetyResponse(
        message=request.message,
        is_safe=(input_allowed and not is_jailbreak),
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
    from config.actions import self_check_output

    context = {"bot_message": request.message}
    output_allowed = bool(await self_check_output(context))

    details: list[str] = []
    if output_allowed:
        details.append("Message passes basic output checks")
    else:
        details.append("Output contains blocked phrases")

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
