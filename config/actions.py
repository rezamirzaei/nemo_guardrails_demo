"""
Custom Actions for NeMo Guardrails

These actions implement the business logic for the guardrails system.
They are called from the Colang flows defined in rails.co
"""

import logging
from typing import Optional
from nemoguardrails.actions import action

logger = logging.getLogger(__name__)


@action(name="self_check_input")
async def self_check_input(context: Optional[dict] = None) -> bool:
    """
    Check if user input is appropriate.
    Returns True if input is allowed, False if it should be blocked.
    """
    user_message = context.get("user_message", "") if context else ""

    # Basic keyword-based checks (supplement LLM-based checks)
    blocked_patterns = [
        "ignore your instructions",
        "forget your rules",
        "pretend you have no",
        "act as if you can",
        "bypass your",
    ]

    message_lower = user_message.lower()
    for pattern in blocked_patterns:
        if pattern in message_lower:
            logger.warning(f"Input blocked due to pattern: {pattern}")
            return False

    return True


@action(name="self_check_output")
async def self_check_output(context: Optional[dict] = None) -> bool:
    """
    Check if bot output is appropriate.
    Returns True if output is allowed, False if it should be blocked.
    """
    bot_message = context.get("bot_message", "") if context else ""

    # Basic checks for inappropriate content
    blocked_phrases = [
        "here's how to hack",
        "here's how to steal",
        "bypass security by",
    ]

    message_lower = bot_message.lower()
    for phrase in blocked_phrases:
        if phrase in message_lower:
            logger.warning(f"Output blocked due to phrase: {phrase}")
            return False

    return True


@action(name="check_jailbreak_attempt")
async def check_jailbreak_attempt(context: Optional[dict] = None) -> bool:
    """
    Detect jailbreak attempts in user input.
    Returns True if a jailbreak is detected, False otherwise.
    """
    user_message = context.get("user_message", "") if context else ""

    jailbreak_indicators = [
        "dan mode",
        "do anything now",
        "ignore all previous",
        "disregard your programming",
        "pretend you are evil",
        "jailbreak",
        "bypass your restrictions",
        "roleplay as an ai without",
        "act like you have no limits",
    ]

    message_lower = user_message.lower()
    for indicator in jailbreak_indicators:
        if indicator in message_lower:
            logger.warning(f"Jailbreak attempt detected: {indicator}")
            return True

    return False


@action(name="check_facts")
async def check_facts(context: Optional[dict] = None) -> bool:
    """
    Basic fact checking for bot responses.
    Returns True if response passes fact check, False if concerns are found.

    Note: In production, this would integrate with external fact-checking
    services or knowledge bases.
    """
    bot_message = context.get("bot_message", "") if context else ""

    # In a real implementation, you would:
    # 1. Extract factual claims from the response
    # 2. Check against a knowledge base or external APIs
    # 3. Flag responses with unverifiable claims

    # For now, we'll pass all responses but log for monitoring
    logger.debug(f"Fact check passed for response: {bot_message[:100]}...")

    return True


@action(name="log_conversation")
async def log_conversation(context: Optional[dict] = None) -> None:
    """
    Log conversation for monitoring and analytics.
    """
    if context:
        user_message = context.get("user_message", "")
        bot_message = context.get("bot_message", "")
        logger.info(f"Conversation logged - User: {user_message[:50]}... Bot: {bot_message[:50]}...")
