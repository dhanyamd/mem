
import asyncio
import os
import redis
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, models
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
COLLECTION_NAME = "cognitive_memory"

async def nuclear_wipe():
    print("🚀 Starting Total Neural Wipe...")
    
    # 1. Wipe Qdrant
    print(f"📡 Connecting to Qdrant at {QDRANT_URL}...")
    client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    if await client.collection_exists(COLLECTION_NAME):
        print(f"🗑️ Deleting Qdrant collection: {COLLECTION_NAME}")
        await client.delete_collection(COLLECTION_NAME)
    
    print(f"🏗️ Re-creating collection: {COLLECTION_NAME}")
    await client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=64, distance=Distance.COSINE)
    )
    
    # Re-apply indices
    print("🔑 Re-applying indices (user_id, user_handle, categories, timestamp)...")
    await client.create_payload_index(COLLECTION_NAME, "user_id", models.PayloadSchemaType.INTEGER)
    await client.create_payload_index(COLLECTION_NAME, "user_handle", models.PayloadSchemaType.KEYWORD)
    await client.create_payload_index(COLLECTION_NAME, "categories", models.PayloadSchemaType.KEYWORD)
    await client.create_payload_index(COLLECTION_NAME, "timestamp", models.PayloadSchemaType.FLOAT)
    
    # 2. Wipe Redis (LMCache)
    print(f"🧹 Connecting to Redis at {REDIS_URL}...")
    try:
        r = redis.Redis.from_url(REDIS_URL)
        # Flush only lmcache keys to be safe, or flushall if it's a dedicated redis
        keys = r.keys("lmcache:*")
        if keys:
            print(f"🔥 Clearing {len(keys)} KV cache entries from Redis...")
            r.delete(*keys)
        else:
            print("✨ Redis KV cache is already empty.")
    except Exception as e:
        print(f"⚠️ Redis wipe failed (is it running?): {e}")

    print("✅ RESET COMPLETE. System is now a Clean Slate.")

if __name__ == "__main__":
    asyncio.run(nuclear_wipe())
