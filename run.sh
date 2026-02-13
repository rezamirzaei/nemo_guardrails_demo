#!/bin/bash
# Build and run script for the NeMo Guardrails application

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

is_true() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

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
if grep -Eq "^GOOGLE_API_KEY=$|^GOOGLE_API_KEY=your-gemini-api-key-here$" .env; then
    if grep -Eq "^LOCAL_LLM_FALLBACK_ENABLED=(1|true|yes|on)$" .env; then
        echo "ℹ️  GOOGLE_API_KEY not set. Using local Ollama fallback."
    else
        echo "⚠️  Set GOOGLE_API_KEY or enable LOCAL_LLM_FALLBACK_ENABLED=true in .env"
        exit 1
    fi
fi

# Load .env variables for startup decisions
set -a
. ./.env
set +a

# Ensure a stable API key when auth is enabled and APP_API_KEY is not explicitly set
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

# Set platform for Mac Intel
export DOCKER_DEFAULT_PLATFORM=linux/amd64

ENABLE_CACHE_RAW="${ENABLE_CACHE:-false}"
ENABLE_CACHE="$(echo "$ENABLE_CACHE_RAW" | tr '[:upper:]' '[:lower:]')"
SERVICES="guardrails-app"
if [[ "$ENABLE_CACHE" =~ ^(1|true|yes|on)$ ]]; then
    export COMPOSE_PROFILES=cache
    SERVICES="guardrails-app redis"
    echo "ℹ️  Cache profile enabled: starting Redis too."
else
    echo "ℹ️  Starting app without Redis (optional cache)."
fi

echo "🔨 Building Docker images..."
docker compose build $SERVICES

echo "🚀 Starting services..."
docker compose up -d $SERVICES

echo ""
echo "✅ Application is starting!"
echo ""
echo "📍 API URL: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "❤️  Health: http://localhost:8000/health"
if [[ "$ENABLE_CACHE" =~ ^(1|true|yes|on)$ ]]; then
    echo "🧠 Redis: http://localhost:6379"
fi
echo ""
echo "📋 View logs: docker compose logs -f"
echo "🛑 Stop: docker compose down"
