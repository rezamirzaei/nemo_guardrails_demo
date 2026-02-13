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
    echo "⚠️  Please edit .env and add your GOOGLE_API_KEY before running again."
    exit 1
fi

# Check if GOOGLE_API_KEY is set
if grep -q "your-gemini-api-key-here" .env; then
    echo "⚠️  Please edit .env and add your actual GOOGLE_API_KEY"
    exit 1
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
