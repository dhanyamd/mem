import asyncio
import numpy as np
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

print("🚀 Initializing Global Embedder (OpenAI: text-embedding-3-small)...")
_CLIENT = AsyncOpenAI()

async def generate_embeddings(strings: list[str], retries: int = 3):
    """
    Generate embeddings using OpenAI with retry logic.
    Returns: np.ndarray of shape (len(strings), 1536)
    """
    if not strings:
        return np.array([])
        
    for attempt in range(retries):
        try:
            resp = await _CLIENT.embeddings.create(
                input=strings,
                model="text-embedding-3-small",
                timeout=15.0 # Explicitly set timeout
            )
            embeddings = [d.embedding for d in resp.data]
            return np.array(embeddings)
        except Exception as e:
            if attempt == retries - 1:
                print(f"❌ OpenAI Embedding failed after {retries} attempts: {e}")
                raise e
            wait = 2 ** attempt
            print(f"⚠️ OpenAI Embedding timeout/error: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)

if __name__ == "__main__":
    texts = ["Hello how are you", "I like Machine learning"]
    res = asyncio.run(generate_embeddings(texts))
    print(f"Embedding shape: {res.shape}")
