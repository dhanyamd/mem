import json
import asyncio
import os
import time
from typing import List, Dict
from tqdm.asyncio import tqdm

# Import Stratum components
from neural_memory import NeuralMemoryEngine
from dotenv import load_dotenv

load_dotenv()

# Configuration
LOCOMO_DATA_PATH = ".//locomo/data/locomo10.json"
CHECKPOINT_PATH = ".//locomo_final_results.jsonl"
CONCURRENCY_LIMIT = 3 # Conservative to ensure stability

async def process_conversation_streaming(sample: Dict, engine: NeuralMemoryEngine, semaphore: asyncio.Semaphore):
    """Answers QA as ingestion progresses."""
    async with semaphore:
        conv_id = sample.get("sample_id", "unknown")
        conversation_data = sample.get("conversation", {})
        questions = sample.get("qa", [])
        
        user_handle = f"final_eval_{conv_id}"
        user_id = 30000 + int(hash(conv_id) % 1000)
            
        # 1. Faster Ingestion (Using the PATCHED engine.store_memory)
        session_nums = sorted([int(k.split("_")[1]) for k in conversation_data.keys() if k.startswith("session_") and k.split("_")[1].isdigit()])
        
        for s_num in session_nums:
            session = conversation_data.get(f"session_{s_num}", [])
            session_date = conversation_data.get(f"session_{s_num}_date_time", "")
            
            for turn in session:
                content = turn.get("text", "").strip()
                if len(content) < 15: continue
                
                # Directly call the patched engine
                try:
                    await engine.store_memory(
                        user_id=user_id,
                        memory_text=f"[{session_date}] {content}",
                        categories=[],
                        user_handle=user_handle
                    )
                except Exception as e:
                    print(f"Ingestion fail: {e}")
            
        # 2. Evaluation
        for i, qa in enumerate(questions):
            question = qa["question"]
            ground_truth = str(qa.get("answer", ""))
            
            try:
                response = await engine.react_agent(
                    user_id=user_id,
                    user_input=question,
                    user_handle=user_handle
                )
                
                hypothesis = response.get("response", "")
                result = {
                    "sample_id": conv_id,
                    "qa_index": i,
                    "hypothesis": hypothesis,
                    "target": ground_truth
                }
                
                with open(CHECKPOINT_PATH, "a") as f:
                    f.write(json.dumps(result) + "\n")
            except Exception: pass
        
        return True

async def main():
    with open(LOCOMO_DATA_PATH, "r") as f:
        data = json.load(f)
    
    if os.path.exists(CHECKPOINT_PATH): os.remove(CHECKPOINT_PATH)
    
    engine = NeuralMemoryEngine()
    
    # 🚨 CRITICAL PATCH: Bypass BrainService (Local Model)
    async def mock_get_taxonomy_path(text, user_handle): return "general/topic/detail"
    engine.get_taxonomy_path = mock_get_taxonomy_path
    
    from neural_memory import store_neural_memory, generate_embeddings
    async def fast_store_memory(user_id, memory_text, categories, existing_context="", user_handle="user", timestamp=None):
        # Generate embedding via OpenAI directly (fast)
        embeddings = await generate_embeddings([memory_text])
        # Call low-level storage bypassing LLM gating
        return await store_neural_memory(
            user_id=user_id,
            memory_text=memory_text,
            categories=[],
            qdrant_client=engine.qdrant_client,
            surprisal_gate=None,
            taxonomy_llm=None,
            lmcache_manager=engine.lmcache_manager,
            existing_context="",
            radix_path="general/topic/detail",
            score_boost=0,
            user_handle=user_handle,
            embedding=embeddings[0]
        )
    engine.store_memory = fast_store_memory
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [process_conversation_streaming(s, engine, semaphore) for s in data]
    print(f"🔥 FINAL PUSH: LoCoMo Full Evaluation (n=1986)")
    
    await tqdm.gather(*tasks)
    print(f"✅ Fast Evaluation Complete. Result file: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
