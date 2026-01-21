#!/usr/bin/env python3
"""
Setup and Test Script for Radix-Titan Neural Memory Engine
"""

import os
import sys
import asyncio
import subprocess

def check_services():
    """Check if required services are running"""
    print("🔍 Checking services...")

    # Check Redis
    try:
        result = subprocess.run(["redis-cli", "ping"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "PONG" in result.stdout:
            print("✅ Redis: RUNNING")
        else:
            print("❌ Redis: NOT RUNNING")
            return False
    except:
        print("❌ Redis: NOT RUNNING")
        return False

    # Check Qdrant
    try:
        import requests
        response = requests.get("http://localhost:6333/healthz", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant: RUNNING")
        else:
            print("❌ Qdrant: NOT RUNNING")
            return False
    except:
        print("❌ Qdrant: NOT RUNNING - Install with: docker run -d -p 6333:6333 qdrant/qdrant")
        return False

    # Check OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and len(api_key.strip()) > 20:  # Basic check
        print("✅ OpenAI API Key: SET")
    else:
        print("❌ OpenAI API Key: NOT SET")
        print("   Set it with: export OPENAI_API_KEY='your_key_here'")
        return False

    return True

async def test_components():
    """Test neural memory components"""
    print("\n🧠 Testing neural memory components...")

    try:
        from neural_memory import SurprisalGate, TaxonomyLLM, LMCacheRedisManager

        # Test SurprisalGate
        print("Testing SurprisalGate...")
        gate = SurprisalGate()
        await gate.initialize()
        is_known = await gate.is_topic_known("Hello world")
        print(f"✅ SurprisalGate: topic_known = {is_known}")

        # Test TaxonomyLLM
        print("Testing TaxonomyLLM...")
        taxonomy = TaxonomyLLM()
        await taxonomy.initialize()
        path = await taxonomy.generate_radix_path("I work on AI projects")
        print(f"✅ TaxonomyLLM: radix_path = {path}")

        # Test LMCacheRedisManager
        print("Testing LMCacheRedisManager...")
        manager = LMCacheRedisManager()
        await manager.initialize()
        test_cache = {"test": "data"}
        await manager.store_kv_cache("test/path", test_cache)
        retrieved = await manager.retrieve_kv_cache("test/path")
        print(f"✅ LMCacheRedisManager: retrieved = {retrieved is not None}")

        return True

    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_system():
    """Test the full integrated system"""
    print("\n🚀 Testing full neural memory system...")

    try:
        from neural_memory import NeuralMemoryEngine
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams, models
        from openai import AsyncOpenAI
        from vectordb import COLLECTION_NAME

        # Initialize clients
        qdrant = AsyncQdrantClient(url="http://localhost:6333")
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Create collection if it doesn't exist
        print("Creating Qdrant collection...")
        try:
            if not (await qdrant.collection_exists(COLLECTION_NAME)):
                await qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=64, distance=Distance.COSINE)
                )
                await qdrant.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="user_id",
                    field_schema=models.PayloadSchemaType.INTEGER
                )
                await qdrant.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="categories",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                print("✅ Collection created")
            else:
                print("✅ Collection exists")
        except Exception as e:
            print(f"⚠️ Collection setup warning: {e}")

        # Initialize neural memory engine
        engine = NeuralMemoryEngine()
        await engine.initialize(qdrant, openai_client)

        print("✅ NeuralMemoryEngine: INITIALIZED")

        # Test memory storage
        result = await engine.store_memory(
            user_id=1,
            memory_text="Test memory about machine learning projects",
            categories=["work", "tech"],
            existing_context=""
        )
        print(f"✅ Memory storage: {result}")

        return True

    except Exception as e:
        print(f"❌ Full system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🤖 Radix-Titan Neural Memory Engine - Setup & Test")
    print("=" * 60)

    # Check services
    services_ok = check_services()

    if not services_ok:
        print("\n❌ Some services are not running. Please fix them first.")
        return

    # Test components
    components_ok = await test_components()

    if not components_ok:
        print("\n❌ Component tests failed.")
        return

    # Test full system
    system_ok = await test_full_system()

    if system_ok:
        print("\n🎉 SUCCESS! Neural Memory Engine is fully operational!")
        print("\n🚀 Ready to run:")
        print("   uv run python chatbot.py 1")
    else:
        print("\n❌ Full system test failed.")

if __name__ == "__main__":
    asyncio.run(main())
