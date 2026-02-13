#!/bin/bash
# Build and run script for the NeMo Guardrails application

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 NeMo Guardrails Docker Application"
echo "======================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your GOOGLE_API_KEY, or keep local fallback enabled."
    exit 1
fi

# Check LLM backend configuration
if grep -Eq "^GOOGLE_API_KEY=(|your-gemini-api-key-here)$" .env; then
    if grep -Eq "^LOCAL_LLM_FALLBACK_ENABLED=(1|true|yes|on)$" .env; then
        echo "ℹ️  GOOGLE_API_KEY not set. Using local Ollama fallback."
    else
        echo "⚠️  Set GOOGLE_API_KEY or enable LOCAL_LLM_FALLBACK_ENABLED=true in .env"
        exit 1
    fi
fi

# Set platform for Mac Intel
export DOCKER_DEFAULT_PLATFORM=linux/amd64

echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "✅ Application is starting!"
echo ""
echo "📍 API URL: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "❤️  Health: http://localhost:8000/health"
echo ""
echo "📋 View logs: docker-compose logs -f"
echo "🛑 Stop: docker-compose down"
