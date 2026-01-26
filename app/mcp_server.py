"""
MCP (Model Context Protocol) Server for NeMo Guardrails

This module implements an MCP server that exposes the guardrails functionality
as tools that can be used by MCP-compatible clients like Claude Desktop.
"""

import os
import logging
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
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


def get_rails() -> LLMRails:
    """Get or initialize the guardrails instance."""
    global rails
    if rails is None:
        logger.info("Loading NeMo Guardrails configuration...")
        config = RailsConfig.from_path("./config")
        rails = LLMRails(config)
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
            rails_instance = get_rails()
            response = await rails_instance.generate_async(
                messages=[{"role": "user", "content": message}]
            )

            response_text = response.get("content", "")
            guardrails_triggered = "I cannot" in response_text or "I'm not able to" in response_text

            result = {
                "response": response_text,
                "guardrails_triggered": guardrails_triggered
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            logger.error(f"Error in chat_with_guardrails: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "check_input_safety":
        message = arguments.get("message", "")
        if not message:
            return [TextContent(type="text", text="Error: message is required")]

        # Import the action functions
        from config.actions import check_jailbreak_attempt, self_check_input

        try:
            context = {"user_message": message}

            is_jailbreak = await check_jailbreak_attempt(context)
            input_allowed = await self_check_input(context)

            result = {
                "message": message,
                "is_safe": input_allowed and not is_jailbreak,
                "jailbreak_detected": is_jailbreak,
                "input_allowed": input_allowed,
                "details": []
            }

            if is_jailbreak:
                result["details"].append("Jailbreak attempt detected")
            if not input_allowed:
                result["details"].append("Input contains blocked patterns")
            if result["is_safe"]:
                result["details"].append("Message passes all safety checks")

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            logger.error(f"Error in check_input_safety: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "get_guardrails_config":
        try:
            config_info = {
                "model": {
                    "engine": "google_genai",
                    "model": os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")
                },
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
