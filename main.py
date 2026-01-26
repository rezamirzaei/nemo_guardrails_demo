"""
NeMo Guardrails Application Entry Point

This file provides a simple entry point for running the application.
The main application code is in app/main.py

Usage:
    # With uv
    uv run python main.py

    # With Docker
    docker-compose up --build

    # Or run the app module directly
    uv run uvicorn app.main:app --reload
"""

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

