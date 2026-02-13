# NeMo Guardrails Docker Application

A production-ready NeMo Guardrails application with Docker support, designed for real-world AI safety implementations. Uses **Google Gemini + Guardrails** when configured, with **local Ollama/Llama fallback** when no API key is provided.

## Features

- **🛡️ Input Rails**: Content moderation and jailbreak detection
- **🔒 Output Rails**: Response validation and fact-checking
- **🔑 API Key Auth**: Secure REST API with authentication
- **💬 Web UI**: AngularJS (MVC-style) chat interface for testing
- **📚 Swagger/ReDoc**: Full API documentation
- **🐳 Docker Support**: Easy deployment with Docker Compose
- **🤖 Dual LLM Backend**: Gemini + Guardrails primary, automatic local Ollama/Llama runtime failover
- **🔌 MCP Support**: Model Context Protocol for Claude Desktop integration
- **💻 Mac Intel Compatible**: Tested on Intel-based Macs (2019+)

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application with Web UI
│   ├── mcp_server.py        # MCP server for Claude Desktop
│   └── static/              # AngularJS UI (index.html, css, js)
├── config/
│   ├── config.yml           # Main guardrails configuration
│   ├── rails.co             # Colang flows (input/output rails)
│   ├── prompts.yml          # Custom prompts for moderation
│   └── actions.py           # Custom Python actions
├── tests/
│   └── test_guardrails.py   # Test suite
├── Dockerfile
├── docker-compose.yml
├── mcp_config.json          # MCP configuration for Claude Desktop
├── pyproject.toml
├── start.sh                 # Quick start script
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template (or create .env)
cp .env.example .env

# Option A: Set GOOGLE_API_KEY for Gemini + Guardrails mode
# Option B: Leave GOOGLE_API_KEY unset and use local Ollama fallback
# For fallback mode, pull a model (example):
# ollama pull llama3.1:8b
```

### 2. Run the Application

**Option A: Using the start script (recommended)**
```bash
./start.sh
```

**Option B: Using uv directly**
```bash
uv run uvicorn app.main:app --reload
```

**Option C: Using Docker**
```bash
docker compose up --build guardrails-app

# Optional: include Redis cache service
docker compose --profile cache up --build
```

### 3. Access the Application

Once running, you'll see an **API key** printed in the console. It is also persisted in `.app_api_key` unless `APP_API_KEY` is explicitly set.

- **🌐 Web UI**: http://localhost:8000 (auto-auth via cookie by default; manual key still supported)
- **📚 Swagger UI**: http://localhost:8000/docs
- **📖 ReDoc**: http://localhost:8000/redoc
- **❤️ Health**: http://localhost:8000/health

### 4. Test with curl

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Chat (requires API key from console)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -d '{"message": "Hello, how can you help me?"}'
```

## MCP (Model Context Protocol) Integration

This application supports MCP for integration with Claude Desktop and other MCP-compatible clients.

### Setup for Claude Desktop

1. Copy the MCP configuration to Claude Desktop's config:

```bash
# Find your Claude Desktop config directory
# macOS: ~/Library/Application Support/Claude/

# Copy the MCP config (or merge with existing config)
cp mcp_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

2. Set `GOOGLE_API_KEY` for Gemini mode, or keep fallback-only mode by enabling local Ollama variables.

   If the `--directory` path in `mcp_config.json` doesn't match your local checkout, update it (or use `http://localhost:8000/mcp/info` to get a ready-to-paste config with the correct path).

3. Restart Claude Desktop.

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `chat_with_guardrails` | Send a message with NeMo Guardrails protection |
| `check_input_safety` | Pre-validate input for safety issues |
| `get_guardrails_config` | Get current guardrails configuration |

### Running MCP Server Standalone

```bash
# Run MCP server directly
uv run python -m app.mcp_server
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI (AngularJS) |
| `/health` | GET | Health check endpoint |
| `/api/info` | GET | API key/authentication info |
| `/api/chat` | POST | Chat with guardrails protection |
| `/api/chat/test` | POST | Mock chat (no LLM call) |
| `/api/tools/check_input_safety` | POST | Lightweight input safety check (no LLM call) |
| `/api/tools/check_output_safety` | POST | Lightweight output safety check (no LLM call) |
| `/api/conversations` | POST | Create a new conversation |
| `/api/conversations/{conversation_id}` | GET | Fetch conversation history |
| `/api/conversations/{conversation_id}` | DELETE | Delete a conversation |
| `/mcp/info` | GET | MCP server configuration info |
| `/docs` | GET | OpenAPI documentation |

### Chat Request

```json
{
  "message": "Your message here",
  "conversation_id": "optional-id"
}
```

### Chat Response

```json
{
  "response": "AI response",
  "conversation_id": "id",
  "guardrails_triggered": false,
  "backend_used": "google_guardrails"
}
```

## Guardrails Configuration

### Input Rails
- **Self Check Input**: Validates user messages against content policy
- **Jailbreak Detection**: Blocks attempts to bypass AI safety measures

### Output Rails
- **Self Check Output**: Ensures responses are appropriate
- **Fact Checking**: Flags potentially inaccurate information

### Custom Actions
Edit `config/actions.py` to add custom business logic:

```python
from nemoguardrails.actions import action

@action(name="custom_check")
async def custom_check(context: dict) -> bool:
    # Your custom logic here
    return True
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | (optional) |
| `GOOGLE_MODEL` | Gemini model to use | gemini-2.0-flash-lite |
| `LOCAL_LLM_FALLBACK_ENABLED` | Enable local Ollama fallback when Gemini is unavailable | true |
| `LOCAL_LLM_MODEL` | Ollama model name | llama3.2:3b |
| `OLLAMA_BASE_URL` | Ollama server URL | http://127.0.0.1:11434 |
| `LOCAL_LLM_TIMEOUT_SECONDS` | Timeout for local LLM calls | 90 |
| `PRIMARY_LLM_TIMEOUT_SECONDS` | Gemini timeout before failover to local model | 20 |
| `APP_API_KEY` | API key for REST API auth | (auto-generated if unset) |
| `API_KEY_REQUIRED` | Require API key auth for `/api/*` | true |
| `UI_AUTH_COOKIE_ENABLED` | Auto-auth browser UI via HttpOnly cookie | true |
| `UI_AUTH_COOKIE_NAME` | Cookie name for browser UI auth | guardrails_ui_api_key |
| `LOG_LEVEL` | Logging level | INFO |
| `DEBUG` | Enable debug mode | false |
| `CORS_ORIGINS` | Comma-separated CORS origins (or `*`) | `*` |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |

## Testing

```bash
# Run tests with uv
uv run pytest tests/ -v

# Run tests with Docker
docker compose run guardrails-app pytest tests/ -v
```

## Mac Intel (2019) Notes

This project is configured for compatibility with Intel-based Macs:

- Docker images use `platform: linux/amd64`
- Python version pinned to 3.10-3.11 for best compatibility
- Dependencies chosen for Intel architecture support

If you encounter issues:

```bash
# Force amd64 platform
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker-compose up --build
```

## Troubleshooting

### Common Issues

1. **Docker build fails on Mac Intel**
   - Ensure Docker Desktop is updated
   - Add `platform: linux/amd64` to compose services

2. **Gemini API errors**
   - Verify your `GOOGLE_API_KEY` is valid
   - Check API quota and billing
   - Or use local fallback with Ollama (`LOCAL_LLM_FALLBACK_ENABLED=true`)
   - If `LOCAL_LLM_MODEL` is missing, the app auto-selects an available local model
   - In Docker, set `OLLAMA_BASE_URL=http://host.docker.internal:11434`
   - Reduce `PRIMARY_LLM_TIMEOUT_SECONDS` if fallback takes too long

3. **Web UI shows authentication failed**
   - Refresh `http://localhost:8000` to renew the UI auth cookie
   - If needed, clear saved browser key and use `cat .app_api_key`
   - Ensure `API_KEY_REQUIRED` and `UI_AUTH_COOKIE_ENABLED` are set as intended

4. **Guardrails not loading**
   - Verify config files are in the `config/` directory
   - Check for YAML syntax errors

### Viewing Logs

```bash
# View container logs
docker compose logs -f guardrails-app

# View specific log level
LOG_LEVEL=DEBUG docker compose up
```

## Production Deployment

For production use:

1. Set `DEBUG=false` in `.env`
2. Set `CORS_ORIGINS` to your allowed origins (do not use `*`)
3. Use secrets management for API keys
4. Enable Redis for caching (already in docker-compose)
5. Add monitoring and alerting

## License

MIT License

## References

- [NeMo Guardrails Documentation](https://docs.nvidia.com/nemo/guardrails/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Gemini API Reference](https://ai.google.dev/api)
- [Ollama Documentation](https://ollama.com/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
