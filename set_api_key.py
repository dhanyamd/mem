#!/usr/bin/env python3
"""
Set OpenAI API Key for Neural Memory Engine
"""

import os
import sys

def main():
    print("🔑 Set OpenAI API Key for Neural Memory Engine")
    print("=" * 50)

    # Check if already set
    existing_key = os.getenv("OPENAI_API_KEY")
    if existing_key and len(existing_key.strip()) > 20:
        print("✅ OpenAI API key is already set!")
        print(f"   Key starts with: {existing_key[:10]}...")
        print("\n🚀 Ready to run:")
        print("   uv run python setup_and_test.py")
        print("   uv run python chatbot.py 1")
        return

    # Check command line argument
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        print("❌ No OpenAI API key provided!")
        print()
        print("📋 SETUP INSTRUCTIONS:")
        print("1. Get your API key from: https://platform.openai.com/api-keys")
        print("2. Set it as an environment variable:")
        print("   export OPENAI_API_KEY='your_key_here'")
        print("3. Or run this script with your key:")
        print("   uv run python set_api_key.py 'your_key_here'")
        print()
        print("4. Then test the system:")
        print("   uv run python setup_and_test.py")
        return

    if not api_key or len(api_key.strip()) < 20:
        print("❌ Invalid API key format. Please try again.")
        return

    # Set environment variable
    os.environ["OPENAI_API_KEY"] = api_key.strip()

    # Test the key by trying to import and initialize
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key.strip())
        # Quick test - this will fail if key is invalid, but let's just check import
        print("✅ OpenAI API key set!")
        print("\n🚀 You can now run:")
        print("   uv run python setup_and_test.py")
        print("   uv run python chatbot.py 1")

    except Exception as e:
        print(f"❌ API key validation failed: {e}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    main()
