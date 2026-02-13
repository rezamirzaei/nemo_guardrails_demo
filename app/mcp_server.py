"""
MCP (Model Context Protocol) Server for NeMo Guardrails

This module implements an MCP server that exposes guardrails functionality
as tools that can be used by MCP-compatible clients like Claude Desktop.
It uses Gemini + NeMo Guardrails when configured, and local Ollama fallback otherwise.
"""

import os
import logging
import json
from pathlib import Path
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)

from dotenv import load_dotenv
from nemoguardrails import RailsConfig, LLMRails

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Create MCP server
mcp_server = Server("nemo-guardrails-mcp")

# Global rails instance
rails: LLMRails | None = None
model_name: str | None = None
backend_mode: str = "unavailable"

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "config"


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _get_non_placeholder_env(name: str, placeholder_values: set[str] | None = None) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        return ""
    normalized = value.lower()
    if placeholder_values and normalized in {p.lower() for p in placeholder_values}:
        return ""
    return value


GOOGLE_API_KEY = _get_non_placeholder_env("GOOGLE_API_KEY", {"your-gemini-api-key-here"})
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-lite").strip() or "gemini-2.0-flash-lite"
LOCAL_LLM_FALLBACK_ENABLED = _get_bool_env("LOCAL_LLM_FALLBACK_ENABLED", True)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.1:8b").strip() or "llama3.1:8b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
try:
    LOCAL_LLM_TIMEOUT_SECONDS = float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "120").strip())
except ValueError:
    LOCAL_LLM_TIMEOUT_SECONDS = 120.0

LLM_BACKEND_GOOGLE = "google_guardrails"
LLM_BACKEND_LOCAL = "local_ollama"
LLM_BACKEND_UNAVAILABLE = "unavailable"

INPUT_REFUSAL_MESSAGE = (
    "I can't help with requests that try to bypass safety controls. "
    "Please rephrase your request."
)
OUTPUT_REFUSAL_MESSAGE = (
    "I can't provide that response because it violates safety checks. "
    "Please ask for a safer alternative."
)


def _is_google_backend_configured() -> bool:
    return bool(GOOGLE_API_KEY)


def _current_backend() -> tuple[str, str]:
    if _is_google_backend_configured():
        return LLM_BACKEND_GOOGLE, model_name or GOOGLE_MODEL
    if LOCAL_LLM_FALLBACK_ENABLED:
        return LLM_BACKEND_LOCAL, LOCAL_LLM_MODEL
    return LLM_BACKEND_UNAVAILABLE, ""


async def _generate_with_ollama(messages: list[dict[str, str]]) -> str:
    timeout = httpx.Timeout(
        timeout=LOCAL_LLM_TIMEOUT_SECONDS,
        connect=min(10.0, LOCAL_LLM_TIMEOUT_SECONDS),
    )
    payload: dict[str, Any] = {
        "model": LOCAL_LLM_MODEL,
        "messages": messages,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
    body = response.json()
    message = body.get("message")
    content = message.get("content") if isinstance(message, dict) else body.get("response")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Local LLM returned an empty response")
    return content.strip()


async def _run_input_safety(message: str) -> tuple[bool, list[str]]:
    from config.actions import check_jailbreak_attempt, self_check_input

    context = {"user_message": message}
    is_jailbreak = bool(await check_jailbreak_attempt(context))
    input_allowed = bool(await self_check_input(context))

    details: list[str] = []
    if is_jailbreak:
        details.append("Jailbreak attempt detected")
    if not input_allowed:
        details.append("Input contains blocked patterns")
    if not details:
        details.append("Message passes basic input checks")

    return input_allowed and not is_jailbreak, details


async def _run_output_safety(message: str) -> tuple[bool, list[str]]:
    from config.actions import check_facts, self_check_output

    output_allowed = bool(await self_check_output({"bot_message": message}))
    facts_ok = bool(await check_facts({"bot_message": message}))

    details: list[str] = []
    if not output_allowed:
        details.append("Output contains blocked phrases")
    if not facts_ok:
        details.append("Output contains unverifiable claims")
    if not details:
        details.append("Message passes basic output checks")

    return output_allowed and facts_ok, details


def get_rails() -> LLMRails:
    """Get or initialize the guardrails instance."""
    global rails, model_name, backend_mode
    if not _is_google_backend_configured():
        raise RuntimeError("GOOGLE_API_KEY is not configured")
    if rails is None:
        logger.info("Loading NeMo Guardrails configuration...")
        config = RailsConfig.from_path(str(CONFIG_DIR))

        for model in config.models:
            if getattr(model, "type", None) == "main":
                model.model = GOOGLE_MODEL

        if model_name is None:
            model_name = next(
                (m.model for m in config.models if getattr(m, "type", None) == "main"),
                None,
            )

        rails = LLMRails(config)
        backend_mode = LLM_BACKEND_GOOGLE
        logger.info("NeMo Guardrails loaded successfully!")
    return rails


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="chat_with_guardrails",
            description="Send a message to the AI assistant with NeMo Guardrails protection. "
                        "Input is moderated for harmful content and jailbreak attempts. "
                        "Output is validated for safety and accuracy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to the AI assistant"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="check_input_safety",
            description="Check if a message would pass the input guardrails without sending it to the LLM. "
                        "Useful for pre-validating user input.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to check for safety"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="get_guardrails_config",
            description="Get information about the current guardrails configuration.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "chat_with_guardrails":
        message = arguments.get("message", "")
        if not message:
            return [TextContent(type="text", text="Error: message is required")]

        try:
            input_is_safe, input_details = await _run_input_safety(message)
            active_backend_mode, active_backend_model = _current_backend()
            guardrails_triggered = False

            if not input_is_safe:
                response_text = INPUT_REFUSAL_MESSAGE
                guardrails_triggered = True
            elif active_backend_mode == LLM_BACKEND_GOOGLE:
                rails_instance = get_rails()
                response = await rails_instance.generate_async(
                    messages=[{"role": "user", "content": message}]
                )
                response_text = str(response.get("content", "")).strip()
            elif active_backend_mode == LLM_BACKEND_LOCAL:
                response_text = await _generate_with_ollama(
                    messages=[{"role": "user", "content": message}]
                )
            else:
                return [
                    TextContent(
                        type="text",
                        text=(
                            "Error: No backend available. Configure GOOGLE_API_KEY "
                            "or enable LOCAL_LLM_FALLBACK_ENABLED with Ollama running."
                        ),
                    )
                ]

            output_is_safe, output_details = await _run_output_safety(response_text)
            if not output_is_safe:
                response_text = OUTPUT_REFUSAL_MESSAGE
                guardrails_triggered = True

            result = {
                "response": response_text,
                "guardrails_triggered": guardrails_triggered,
                "backend_mode": active_backend_mode,
                "backend_model": active_backend_model,
                "input_checks": input_details,
                "output_checks": output_details,
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            logger.error(f"Error in chat_with_guardrails: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "check_input_safety":
        message = arguments.get("message", "")
        if not message:
            return [TextContent(type="text", text="Error: message is required")]

        try:
            is_safe, details = await _run_input_safety(message)
            jailbreak_detected = "Jailbreak attempt detected" in details
            input_allowed = "Input contains blocked patterns" not in details

            result = {
                "message": message,
                "is_safe": is_safe,
                "jailbreak_detected": jailbreak_detected,
                "input_allowed": input_allowed,
                "details": details,
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            logger.error(f"Error in check_input_safety: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "get_guardrails_config":
        try:
            active_backend_mode, active_backend_model = _current_backend()
            config_info = {
                "model": {
                    "engine": "google_genai" if active_backend_mode == LLM_BACKEND_GOOGLE else "ollama",
                    "model": active_backend_model or "none",
                },
                "backend_mode": active_backend_mode,
                "rails": {
                    "input": ["self_check_input", "check_jailbreak"],
                    "output": ["self_check_output", "check_facts"]
                },
                "features": [
                    "Content moderation",
                    "Jailbreak detection",
                    "Output validation",
                    "Fact checking"
                ]
            }

            return [TextContent(type="text", text=json.dumps(config_info, indent=2))]

        except Exception as e:
            logger.error(f"Error in get_guardrails_config: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_mcp_server():
    """Run the MCP server using stdio transport."""
    logger.info("Starting MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_mcp_server())
