#!/usr/bin/env python3
"""
Simple test for the neural memory chat system
"""

import asyncio
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_chat():
    try:
        print("🧠 Testing Neural Memory Chat System...")

        from response_generator import react_agent, get_all_categories
        from neural_memory import NeuralMemoryEngine
        from qdrant_client import AsyncQdrantClient

        print("✅ Imports successful")

        # Set API key
        # os.environ['OPENAI_API_KEY'] = 'sk-proj-...'

        # Initialize clients
        qdrant = AsyncQdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        neural_memory_engine = NeuralMemoryEngine()
        await neural_memory_engine.initialize(qdrant, None)

        existing_categories = await get_all_categories(user_id=1)
        print(f"✅ Found {len(existing_categories)} categories")

        # Test conversation
        past_messages = []
        question = 'Hello, tell me about neural networks.'

        print(f"\n🤖 User: {question}")

        result = await react_agent(
            user_id=1,
            transcript=past_messages,
            question=question,
            existing_categories=existing_categories,
            tools={},
            neural_memory_engine=neural_memory_engine,
            max_iters=1
        )

        print(f"💬 AI: {result.response[:100]}...")
        print(f"💾 Memory saved: {result.save_memory}")

        print("\n🎉 SUCCESS: Neural Memory Chat System Working!")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_chat())
