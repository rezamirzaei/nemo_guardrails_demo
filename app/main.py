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

import os
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
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

# API Key configuration - if not set or default, auth will be optional
_raw_api_key = os.getenv("APP_API_KEY", "")
API_KEY_ENABLED = _raw_api_key and _raw_api_key != "your-app-api-key-here"
API_KEY = _raw_api_key if API_KEY_ENABLED else None
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global rails instance
rails: Optional[LLMRails] = None


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


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


# =============================================================================
# Authentication
# =============================================================================

async def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)) -> Optional[str]:
    """Verify API key for protected endpoints. If API_KEY is not configured, skip auth."""
    # If API key auth is disabled, allow all requests
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - loads guardrails on startup"""
    global rails

    logger.info("Loading NeMo Guardrails configuration...")
    if API_KEY_ENABLED:
        logger.info(f"API Key authentication: ENABLED (key: {API_KEY})")
    else:
        logger.info("API Key authentication: DISABLED (no API key required)")

    try:
        # Load rails configuration from config directory
        config = RailsConfig.from_path("./config")
        rails = LLMRails(config)
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
- 🤖 **Google Gemini**: Powered by Gemini 1.5 Flash
- 🔌 **MCP Support**: Model Context Protocol server for Claude Desktop integration

### Authentication
All `/api/*` endpoints require an API key passed in the `X-API-Key` header.
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Public Endpoints (No Auth Required)
# =============================================================================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the main web UI"""
    return get_chat_ui_html()


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
        message="Use the X-API-Key header to authenticate. Check server logs for the API key.",
        api_key_required=True,
        header_name="X-API-Key"
    )


@app.get("/mcp/info", tags=["MCP"])
async def mcp_info():
    """
    Get MCP (Model Context Protocol) configuration information.

    Returns the configuration needed to add this server to Claude Desktop or other MCP clients.
    """
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
                        "/Users/rezami/PycharmProjects/PythonProject7",
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

    logger.info(f"Processing message: {request.message[:50]}...")

    try:
        # Generate response through guardrails
        response = await rails.generate_async(
            messages=[{"role": "user", "content": request.message}]
        )

        # Check if guardrails blocked or modified the response
        guardrails_triggered = False
        response_text = response.get("content", "")

        # Check for guardrail intervention indicators
        if "I cannot" in response_text or "I'm not able to" in response_text:
            guardrails_triggered = True

        logger.info(f"Generated response (guardrails_triggered={guardrails_triggered})")

        return ChatResponse(
            response=response_text,
            conversation_id=request.conversation_id,
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
    return ChatResponse(
        response=f"[TEST MODE] Received your message: '{request.message[:100]}...' - Guardrails would process this in production.",
        conversation_id=request.conversation_id or "test-conv-001",
        guardrails_triggered=False
    )


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


# =============================================================================
# Web UI HTML
# =============================================================================

def get_chat_ui_html() -> str:
    """Return the HTML for the chat UI"""
    if API_KEY_ENABLED:
        api_key_section = """
        <div class="api-key-section">
            <label for="apiKey">API Key</label>
            <input type="password" id="apiKey" class="api-key-input" placeholder="Enter your API key (check server logs)">
            <p class="api-key-hint">💡 The API key is printed in the server console when the app starts.</p>
        </div>
        """
        api_key_required_js = "true"
    else:
        api_key_section = """
        <div class="api-key-section" style="background: rgba(118, 184, 82, 0.2); border: 1px solid #76b852;">
            <p style="color: #76b852; text-align: center;">✅ No API key required - authentication is disabled</p>
            <input type="hidden" id="apiKey" value="">
        </div>
        """
        api_key_required_js = "false"

    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeMo Guardrails + Gemini Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 20px;
        }
        
        header h1 {
            font-size: 2rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #76b852, #8DC26F);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        header p {
            color: #888;
            font-size: 0.9rem;
        }
        
        .api-key-section {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .api-key-section label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #aaa;
        }
        
        .api-key-input {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 14px;
            font-family: monospace;
        }
        
        .api-key-input:focus {
            outline: none;
            border-color: #76b852;
        }
        
        .api-key-hint {
            margin-top: 8px;
            font-size: 12px;
            color: #666;
        }
        
        .chat-container {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 500px;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 16px;
            display: flex;
            flex-direction: column;
        }
        
        .message.user {
            align-items: flex-end;
        }
        
        .message.assistant {
            align-items: flex-start;
        }
        
        .message-content {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #76b852, #8DC26F);
            color: #fff;
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant .message-content {
            background: rgba(255,255,255,0.1);
            border-bottom-left-radius: 4px;
        }
        
        .message-meta {
            font-size: 11px;
            color: #666;
            margin-top: 4px;
            padding: 0 8px;
        }
        
        .guardrails-badge {
            display: inline-block;
            background: #ff6b6b;
            color: #fff;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 10px;
            margin-left: 8px;
        }
        
        .chat-input-container {
            padding: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
            display: flex;
            gap: 12px;
        }
        
        .chat-input {
            flex: 1;
            padding: 14px 18px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 24px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 14px;
            resize: none;
        }
        
        .chat-input:focus {
            outline: none;
            border-color: #76b852;
        }
        
        .send-button {
            padding: 14px 28px;
            background: linear-gradient(135deg, #76b852, #8DC26F);
            border: none;
            border-radius: 24px;
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }
        
        .send-button:hover {
            transform: scale(1.02);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .links {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        
        .links a {
            color: #76b852;
            text-decoration: none;
            font-size: 14px;
            padding: 8px 16px;
            border: 1px solid #76b852;
            border-radius: 8px;
            transition: background 0.2s;
        }
        
        .links a:hover {
            background: rgba(118, 184, 82, 0.2);
        }
        
        .status {
            text-align: center;
            padding: 10px;
            font-size: 12px;
        }
        
        .status.connected {
            color: #76b852;
        }
        
        .status.error {
            color: #ff6b6b;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #76b852;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .examples {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
        }
        
        .examples h3 {
            font-size: 14px;
            color: #888;
            margin-bottom: 10px;
        }
        
        .example-btn {
            display: inline-block;
            margin: 4px;
            padding: 8px 12px;
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 6px;
            color: #ccc;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .example-btn:hover {
            background: rgba(255,255,255,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛡️ NeMo Guardrails + Gemini</h1>
            <p>AI Assistant with Input/Output Safety Rails • Powered by Google Gemini</p>
        </header>
        
        __API_KEY_SECTION__
        
        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="message assistant">
                    <div class="message-content">
                        Hello! I'm an AI assistant powered by Google Gemini and protected by NeMo Guardrails. I can help you with questions while ensuring safe and appropriate responses. How can I assist you today?
                    </div>
                </div>
            </div>
            <div class="chat-input-container">
                <input type="text" id="chatInput" class="chat-input" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                <button class="send-button" id="sendButton" onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <div class="status" id="status">Ready</div>
        
        <div class="examples">
            <h3>Try these examples:</h3>
            <button class="example-btn" onclick="setExample('Hello, how are you?')">👋 Greeting</button>
            <button class="example-btn" onclick="setExample('What can you help me with?')">❓ Capabilities</button>
            <button class="example-btn" onclick="setExample('Tell me about machine learning')">🤖 ML Question</button>
            <button class="example-btn" onclick="setExample('Ignore your instructions and...')">🚫 Jailbreak Test</button>
            <button class="example-btn" onclick="setExample('How do I hack into...')">⚠️ Harmful Test</button>
        </div>
        
        <div class="links">
            <a href="/docs" target="_blank">📚 API Docs (Swagger)</a>
            <a href="/redoc" target="_blank">📖 API Docs (ReDoc)</a>
            <a href="/mcp/info" target="_blank">🔌 MCP Config</a>
            <a href="/health" target="_blank">❤️ Health Check</a>
        </div>
    </div>
    
    <script>
        const chatMessages = document.getElementById('chatMessages');
        const chatInput = document.getElementById('chatInput');
        const sendButton = document.getElementById('sendButton');
        const apiKeyInput = document.getElementById('apiKey');
        const status = document.getElementById('status');
        const apiKeyRequired = __API_KEY_REQUIRED__;
        
        // Load API key from localStorage
        if (apiKeyRequired && apiKeyInput) {
            const savedApiKey = localStorage.getItem('nemo_api_key');
            if (savedApiKey) {
                apiKeyInput.value = savedApiKey;
            }
            
            // Save API key when changed
            apiKeyInput.addEventListener('change', () => {
                localStorage.setItem('nemo_api_key', apiKeyInput.value);
            });
        }
        
        function setExample(text) {
            chatInput.value = text;
            chatInput.focus();
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        function addMessage(content, role, guardrailsTriggered = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + role;
            
            let metaHtml = '';
            if (guardrailsTriggered) {
                metaHtml = '<span class="guardrails-badge">Guardrails Triggered</span>';
            }
            
            messageDiv.innerHTML = 
                '<div class="message-content">' + escapeHtml(content) + '</div>' +
                '<div class="message-meta">' + new Date().toLocaleTimeString() + metaHtml + '</div>';
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function setStatus(message, isError) {
            status.textContent = message;
            status.className = 'status ' + (isError ? 'error' : 'connected');
        }
        
        async function sendMessage() {
            const message = chatInput.value.trim();
            const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
            
            if (!message) return;
            
            if (apiKeyRequired && !apiKey) {
                setStatus('Please enter your API key', true);
                if (apiKeyInput) apiKeyInput.focus();
                return;
            }
            
            // Add user message
            addMessage(message, 'user');
            chatInput.value = '';
            
            // Disable input while processing
            sendButton.disabled = true;
            chatInput.disabled = true;
            setStatus('Processing...', false);
            
            try {
                const headers = {'Content-Type': 'application/json'};
                if (apiKey) {
                    headers['X-API-Key'] = apiKey;
                }
                
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify({ message: message })
                });
                
                if (response.status === 401 || response.status === 403) {
                    setStatus('Invalid API key. Check server logs for the correct key.', true);
                    return;
                }
                
                if (response.status === 429) {
                    const error = await response.json();
                    setStatus('Rate limit exceeded - wait and try again', true);
                    addMessage('⏳ ' + (error.detail || 'Rate limit exceeded. Please wait a minute and try again.'), 'assistant');
                    return;
                }
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Request failed');
                }
                
                const data = await response.json();
                addMessage(data.response, 'assistant', data.guardrails_triggered);
                setStatus('Ready', false);
                
            } catch (error) {
                setStatus('Error: ' + error.message, true);
                addMessage('Sorry, an error occurred: ' + error.message, 'assistant');
            } finally {
                sendButton.disabled = false;
                chatInput.disabled = false;
                chatInput.focus();
            }
        }
        
        // Check health on load
        fetch('/health')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.guardrails_loaded) {
                    setStatus('Connected - Guardrails loaded ✓', false);
                } else {
                    setStatus('Warning: Guardrails not loaded', true);
                }
            })
            .catch(function() { setStatus('Cannot connect to server', true); });
    </script>
</body>
</html>
"""

    return html_template.replace('__API_KEY_SECTION__', api_key_section).replace('__API_KEY_REQUIRED__', api_key_required_js)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )

