#!/bin/bash
# Quick setup and run script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file"
fi

# Check backend configuration
if grep -Eq "^GOOGLE_API_KEY=(|your-gemini-api-key-here)$" .env 2>/dev/null; then
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

echo "Starting NeMo Guardrails server..."
echo ""
echo "📍 Web UI:      http://localhost:8000"
echo "📚 Swagger UI:  http://localhost:8000/docs"
echo "📖 ReDoc:       http://localhost:8000/redoc"
echo "🔌 MCP Config:  http://localhost:8000/mcp/info"
echo "❤️  Health:      http://localhost:8000/health"
echo ""
echo "🔑 The API key will be printed below when the server starts."
echo "   Copy it and paste it into the Web UI to authenticate."
echo ""

# Run the server
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
