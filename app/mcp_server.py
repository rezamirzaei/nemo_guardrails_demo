"""
MCP (Model Context Protocol) Server for NeMo Guardrails

This module implements an MCP server that exposes guardrails functionality
as tools that can be used by MCP-compatible clients like Claude Desktop.
It uses Gemini + NeMo Guardrails when configured, and local Ollama fallback otherwise.
"""

import os
import logging
import json
import asyncio
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
    LOCAL_LLM_TIMEOUT_SECONDS = float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "90").strip())
except ValueError:
    LOCAL_LLM_TIMEOUT_SECONDS = 90.0
try:
    PRIMARY_LLM_TIMEOUT_SECONDS = float(os.getenv("PRIMARY_LLM_TIMEOUT_SECONDS", "20").strip())
except ValueError:
    PRIMARY_LLM_TIMEOUT_SECONDS = 20.0

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


async def _list_ollama_models(client: httpx.AsyncClient | None = None) -> list[str]:
    async def _fetch(async_client: httpx.AsyncClient) -> list[str]:
        response = await async_client.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models")
        if not isinstance(models, list):
            return []
        names: list[str] = []
        for model in models:
            if not isinstance(model, dict):
                continue
            name = model.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return names

    try:
        if client is not None:
            return await _fetch(client)
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as async_client:
            return await _fetch(async_client)
    except Exception:
        return []


def _select_best_local_model(available_models: list[str]) -> str | None:
    if not available_models:
        return None

    preferred = [
        LOCAL_LLM_MODEL,
        "llama3.2:3b",
        "llama3.1:8b",
        "llama3.2:1b",
        "phi3:mini",
    ]
    available_lower = {m.lower(): m for m in available_models}
    for model in preferred:
        selected = available_lower.get(model.lower())
        if selected:
            return selected
    return available_models[0]


async def _generate_with_ollama(messages: list[dict[str, str]]) -> str:
    model_to_use = LOCAL_LLM_MODEL
    timeout = httpx.Timeout(
        timeout=LOCAL_LLM_TIMEOUT_SECONDS,
        connect=min(10.0, LOCAL_LLM_TIMEOUT_SECONDS),
    )
    payload: dict[str, Any] = {
        "model": model_to_use,
        "messages": messages,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        if response.status_code == 404:
            try:
                err = str(response.json().get("error", "")).lower()
            except Exception:
                err = response.text.lower()
            if "model" in err and "not found" in err:
                available_models = await _list_ollama_models(client)
                fallback_model = _select_best_local_model(available_models)
                if fallback_model and fallback_model.lower() != model_to_use.lower():
                    logger.warning(
                        f"Ollama model '{model_to_use}' not found. "
                        f"Retrying with '{fallback_model}'."
                    )
                    model_to_use = fallback_model
                    payload["model"] = model_to_use
                    response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                else:
                    available_text = ", ".join(available_models) if available_models else "(none)"
                    raise ValueError(
                        f"Ollama model '{LOCAL_LLM_MODEL}' is not available. "
                        f"Available models: {available_text}."
                    )
        response.raise_for_status()
    body = response.json()
    message = body.get("message")
    content = message.get("content") if isinstance(message, dict) else body.get("response")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Local LLM returned an empty response")
    return content.strip()


def _action_result_to_bool(result: Any) -> bool | None:
    if isinstance(result, bool):
        return result

    return_value = getattr(result, "return_value", None)
    if isinstance(return_value, bool):
        return return_value

    return None


async def _run_guardrails_action_bool(action_name: str, context: dict[str, str]) -> bool | None:
    if not _is_google_backend_configured():
        return None

    try:
        rails_instance = get_rails()
    except Exception:
        return None

    runtime = getattr(rails_instance, "runtime", None)
    if runtime is None:
        return None

    action_dispatcher = getattr(runtime, "action_dispatcher", None)
    llm_task_manager = getattr(runtime, "llm_task_manager", None)
    llm = getattr(rails_instance, "llm", None)
    if action_dispatcher is None or llm_task_manager is None or llm is None:
        return None

    if not action_dispatcher.has_registered(action_name):
        return None

    params: dict[str, Any] = {
        "context": context,
        "llm_task_manager": llm_task_manager,
        "llm": llm,
        "config": rails_instance.config,
    }
    result, status = await action_dispatcher.execute_action(action_name, params)
    if status != "success":
        return None
    return _action_result_to_bool(result)


async def _run_input_safety(message: str) -> tuple[bool, list[str]]:
    from config.actions import check_jailbreak_attempt, keyword_input_filter

    context = {"user_message": message}
    is_jailbreak = bool(await check_jailbreak_attempt(context))
    keyword_allowed = bool(await keyword_input_filter(context))

    semantic_allowed = True
    semantic_result = await _run_guardrails_action_bool("self_check_input", context)
    if semantic_result is not None:
        semantic_allowed = bool(semantic_result)

    input_allowed = keyword_allowed and semantic_allowed

    details: list[str] = []
    if is_jailbreak:
        details.append("Jailbreak attempt detected")
    if not keyword_allowed:
        details.append("Input blocked by keyword prefilter")
    if not semantic_allowed:
        details.append("Input blocked by NeMo self-check")
    if not details:
        details.append("Message passes basic input checks")

    return input_allowed and not is_jailbreak, details


async def _run_output_safety(message: str) -> tuple[bool, list[str]]:
    from config.actions import check_facts, keyword_output_filter

    keyword_allowed = bool(await keyword_output_filter({"bot_message": message}))
    semantic_allowed = True
    semantic_result = await _run_guardrails_action_bool("self_check_output", {"bot_message": message})
    if semantic_result is not None:
        semantic_allowed = bool(semantic_result)

    output_allowed = keyword_allowed and semantic_allowed
    facts_ok = bool(await check_facts({"bot_message": message}))

    details: list[str] = []
    if not keyword_allowed:
        details.append("Output blocked by keyword prefilter")
    if not semantic_allowed:
        details.append("Output blocked by NeMo self-check")
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
                try:
                    rails_instance = get_rails()
                    response = await asyncio.wait_for(
                        rails_instance.generate_async(
                            messages=[{"role": "user", "content": message}]
                        ),
                        timeout=PRIMARY_LLM_TIMEOUT_SECONDS,
                    )
                    response_text = str(response.get("content", "")).strip()
                    if not response_text:
                        raise ValueError("Gemini returned an empty response")
                except Exception as exc:
                    if not LOCAL_LLM_FALLBACK_ENABLED:
                        raise
                    logger.warning(
                        f"Gemini request failed in MCP tool ({type(exc).__name__}: {exc}); "
                        "falling back to local Ollama."
                    )
                    response_text = await _generate_with_ollama(
                        messages=[{"role": "user", "content": message}]
                    )
                    active_backend_mode = LLM_BACKEND_LOCAL
                    active_backend_model = LOCAL_LLM_MODEL
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
