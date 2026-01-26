#!/usr/bin/env python3
"""
Test script to verify Google Gemini API connectivity.
Run this to troubleshoot API issues.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if API key is set."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY is not set!")
        return False
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    return True

def test_direct_api():
    """Test direct Google GenAI API."""
    print("\n🔍 Testing direct Google GenAI API...")
    try:
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)

        # List available models
        print("\n📋 Available models:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"  - {model.name}")

        # Test with gemini-2.0-flash-lite (available model)
        print("\n💬 Testing gemini-2.0-flash-lite...")
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = model.generate_content("Say 'Hello from Gemini!' in exactly 5 words")
        print(f"✅ Response: {response.text}")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_nemo_guardrails():
    """Test NeMo Guardrails with the config."""
    print("\n🛡️ Testing NeMo Guardrails...")
    try:
        from nemoguardrails import RailsConfig, LLMRails

        print("  Loading config from ./config ...")
        config = RailsConfig.from_path("./config")
        print("  Creating LLMRails instance...")
        rails = LLMRails(config)
        print("  ✅ Guardrails loaded successfully!")

        print("\n💬 Testing chat through guardrails...")
        import asyncio

        async def test_chat():
            response = await rails.generate_async(
                messages=[{"role": "user", "content": "Hello, how are you?"}]
            )
            return response

        response = asyncio.run(test_chat())
        print(f"✅ Response: {response.get('content', 'No content')}")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 NeMo Guardrails + Google Gemini Test")
    print("=" * 60)

    all_passed = True

    if not test_api_key():
        all_passed = False
        sys.exit(1)

    if not test_direct_api():
        all_passed = False

    if not test_nemo_guardrails():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check the errors above.")
    print("=" * 60)
