"""
Custom Actions for NeMo Guardrails

These actions implement the business logic for the guardrails system.
They are called from the Colang flows defined in rails.co
"""

import logging
import re
from difflib import SequenceMatcher
from typing import Optional

from nemoguardrails.actions import action

logger = logging.getLogger(__name__)


INPUT_BLOCK_PATTERNS = (
    "ignore your instructions",
    "forget your rules",
    "pretend you have no",
    "act as if you can",
    "bypass your",
)

OUTPUT_BLOCK_PHRASES = (
    "here's how to hack",
    "here's how to steal",
    "bypass security by",
)

JAILBREAK_INDICATORS = (
    "dan mode",
    "do anything now",
    "ignore all previous",
    "ignore previous instructions",
    "disregard your programming",
    "pretend you are evil",
    "jailbreak",
    "bypass your restrictions",
    "roleplay as an ai without",
    "act like you have no limits",
)

JAILBREAK_REGEXES = (
    r"\bign\w*\s+all\s+prev\w*\b",
    r"\bignore\b.{0,25}\b(instruction|rule|policy|guideline)s?\b",
    r"\b(bypass|disable|override)\b.{0,25}\b(safety|guardrail|restriction|policy)\b",
)


def _normalize_message(text: str) -> str:
    lowered = text.lower()
    # Keep alnum + whitespace; normalize punctuation/extra spacing.
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _fuzzy_contains(normalized_message: str, normalized_indicator: str, threshold: float) -> bool:
    if not normalized_message or not normalized_indicator:
        return False

    if normalized_indicator in normalized_message:
        return True

    words = normalized_message.split()
    indicator_words = normalized_indicator.split()
    window_size = len(indicator_words)
    if window_size == 0 or len(words) < window_size:
        return False

    for idx in range(len(words) - window_size + 1):
        window = " ".join(words[idx : idx + window_size])
        if SequenceMatcher(None, window, normalized_indicator).ratio() >= threshold:
            return True

    return False


def _matches_phrase(message: str, phrase: str, threshold: float = 0.87) -> bool:
    normalized_message = _normalize_message(message)
    normalized_phrase = _normalize_message(phrase)
    return _fuzzy_contains(normalized_message, normalized_phrase, threshold)


@action(name="keyword_input_filter")
async def keyword_input_filter(context: Optional[dict] = None) -> bool:
    """
    Fast deterministic prefilter for user input.
    Returns True if input is allowed, False if it should be blocked.
    """
    user_message = context.get("user_message", "") if context else ""
    for pattern in INPUT_BLOCK_PATTERNS:
        if _matches_phrase(user_message, pattern):
            logger.warning(f"Input blocked due to keyword prefilter: {pattern}")
            return False

    return True


@action(name="keyword_output_filter")
async def keyword_output_filter(context: Optional[dict] = None) -> bool:
    """
    Fast deterministic prefilter for bot output.
    Returns True if output is allowed, False if it should be blocked.
    """
    bot_message = context.get("bot_message", "") if context else ""
    for phrase in OUTPUT_BLOCK_PHRASES:
        if _matches_phrase(bot_message, phrase, threshold=0.9):
            logger.warning(f"Output blocked due to keyword prefilter: {phrase}")
            return False

    return True


@action(name="check_jailbreak_attempt")
async def check_jailbreak_attempt(context: Optional[dict] = None) -> bool:
    """
    Detect jailbreak attempts in user input.
    Returns True if a jailbreak is detected, False otherwise.
    """
    user_message = context.get("user_message", "") if context else ""
    normalized_message = _normalize_message(user_message)

    for regex in JAILBREAK_REGEXES:
        if re.search(regex, normalized_message):
            logger.warning(f"Jailbreak attempt detected by regex: {regex}")
            return True

    for indicator in JAILBREAK_INDICATORS:
        if _matches_phrase(user_message, indicator, threshold=0.84):
            logger.warning(f"Jailbreak attempt detected by indicator: {indicator}")
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
