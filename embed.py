import asyncio
import openai
import numpy as np 
import os 

client = openai.AsyncClient(api_key=os.getenv("OPENAI_API_KEY")) 

async def generate_embeddings(strings: list[str]): 
    out = await client.embeddings.create(
        input=strings,
        model="text-embedding-3-small",
        dimensions=64
    )
    embeddings = np.stack([item.embedding for item in out.data]) 
    return embeddings 

if __name__ == "__main__": 
    texts = [
        "Hello how are you",
        "I like Machine learning"
    ] 
    asyncio.run(generate_embeddings(texts))
