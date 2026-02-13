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

# Check if GOOGLE_API_KEY is set
if grep -q "your-gemini-api-key-here" .env 2>/dev/null; then
    echo ""
    echo "⚠️  IMPORTANT: You need to set your GOOGLE_API_KEY in .env"
    echo "   Edit the file: .env"
    echo "   Get your key from: https://makersuite.google.com/app/apikey"
    echo ""
fi

echo "Starting NeMo Guardrails + Gemini server..."
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
