#!/bin/bash
# Quick setup and run script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

is_true() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file"
fi

# Check backend configuration
if grep -Eq "^GOOGLE_API_KEY=$|^GOOGLE_API_KEY=your-gemini-api-key-here$" .env 2>/dev/null; then
    if grep -Eq "^LOCAL_LLM_FALLBACK_ENABLED=(1|true|yes|on)$" .env 2>/dev/null; then
        echo ""
        echo "ℹ️  GOOGLE_API_KEY not set. Server will use local Ollama fallback."
        echo "   Make sure Ollama is running and your LOCAL_LLM_MODEL is available."
        echo ""
    else
        echo ""
        echo "⚠️  IMPORTANT: Set GOOGLE_API_KEY or enable LOCAL_LLM_FALLBACK_ENABLED=true in .env"
        echo ""
        exit 1
    fi
fi

# Load .env for a stable local API key
set -a
. ./.env
set +a

if is_true "${API_KEY_REQUIRED:-true}"; then
    if [ -z "${APP_API_KEY:-}" ] || [ "${APP_API_KEY}" = "your-app-api-key-here" ]; then
        if [ -s ".app_api_key" ]; then
            APP_API_KEY="$(tr -d '\r\n' < .app_api_key)"
        else
            APP_API_KEY="$(openssl rand -hex 24)"
            printf "%s\n" "$APP_API_KEY" > .app_api_key
            chmod 600 .app_api_key || true
        fi
        export APP_API_KEY
        echo "🔑 API key: $APP_API_KEY"
        echo "   Stored in .app_api_key"
    else
        echo "🔑 API key: $APP_API_KEY"
        echo "   Loaded from APP_API_KEY in .env"
    fi
else
    echo "🔓 API key auth is disabled (API_KEY_REQUIRED=false)."
fi

echo "Starting NeMo Guardrails server..."
echo ""
echo "📍 Web UI:      http://localhost:8000"
echo "📚 Swagger UI:  http://localhost:8000/docs"
echo "📖 ReDoc:       http://localhost:8000/redoc"
echo "🔌 MCP Config:  http://localhost:8000/mcp/info"
echo "❤️  Health:      http://localhost:8000/health"
echo ""
echo "Paste the API key above into the Web UI."
echo ""

# Run the server
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
