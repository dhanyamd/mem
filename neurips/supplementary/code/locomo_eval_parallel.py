import json
import asyncio
import os
import time
from typing import List, Dict
from tqdm.asyncio import tqdm

# Import Stratum components
from neural_memory import NeuralMemoryEngine
from vectordb import create_memory_collection
from dotenv import load_dotenv

load_dotenv()

# Configuration
LOCOMO_DATA_PATH = ".//locomo/data/locomo10.json"
CHECKPOINT_PATH = ".//locomo_FINAL_GOLDEN.jsonl"
CONCURRENCY_LIMIT = 1  # Snail-Titan Mode (Guaranteed Finish)
COOLDOWN_SECONDS = 10   # Rest period to reset API quotas

async def process_conversation(sample: Dict, engine: NeuralMemoryEngine, conv_semaphore: asyncio.Semaphore, total_semaphore: asyncio.Semaphore):
    """Ingests and evaluates a single conversation history and its questions."""
    async with conv_semaphore:
        conv_id = sample.get("sample_id", "unknown")
        conversation_data = sample.get("conversation", {})
        questions = sample.get("qa", [])
        
        user_handle = f"loco_{conv_id}"
        # Fixed Mapping from Cloud Qdrant Facets
        HANDLE_ID_MAP = {
            "loco_conv-50": 9284,
            "loco_conv-44": 9517,
            "loco_conv-26": 9189,
            "loco_conv-47": 9616,
            "loco_conv-30": 9973,
            "loco_conv-43": 9460,
            "loco_conv-49": 9447,
            "loco_conv-42": 9592,
            "loco_conv-48": 9477,
            "loco_conv-41": 9739
        }
        user_id = HANDLE_ID_MAP.get(user_handle, 9999)
            
        # Check if already ingested (Skip Ingestion if > 100 points exist for this handle)
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        count_res = await engine.qdrant_client.count(
            collection_name="cognitive_memory",
            count_filter=Filter(must=[
                FieldCondition(key="user_handle", match=MatchValue(value=user_handle))
            ])
        )
        if count_res.count > 100:
            pass # Keep ingestion skipped if points exist
        else:
            # 1. Ingestion
            running_context = ""
            # Get all session numbers by looking for 'session_N' keys
            session_nums = sorted([int(k.split("_")[1]) for k in conversation_data.keys() if k.startswith("session_") and k.split("_")[1].isdigit()])
            
            for s_num in session_nums:
                session = conversation_data.get(f"session_{s_num}", [])
                session_date = conversation_data.get(f"session_{s_num}_date_time", "")
                
                for turn in session:
                    content = turn.get("text", "").strip()
                    speaker = turn.get("speaker", "")
                    if len(content) < 10: continue
                    
                    memory_text = f"[{session_date}] {speaker}: {content}" if session_date else f"{speaker}: {content}"
                    
                    try:
                        await engine.store_memory(
                            user_id=user_id,
                            memory_text=memory_text,
                            categories=[],
                            user_handle=user_handle,
                            existing_context=running_context
                        )
                        running_context = (running_context + "\n" + memory_text)[-1500:]
                    except Exception: continue
            
        # 2. Evaluation (QA)
        results = []
        for i, qa in enumerate(questions):
            async with total_semaphore:
                question = qa["question"]
                ground_truth = str(qa.get("answer", ""))
                category = qa.get("category", 0)
                
                try:
                    # Lean Pipeline: Direct Delegation to react_agent
                    from response_generator import react_agent
                    
                    transcript = [{"role": "user", "content": question}]
                    hypothesis = ""
                    
                    async for chunk in react_agent(user_id, transcript, question, engine=engine, user_handle=user_handle):
                        if chunk["type"] == "final_output":
                            hypothesis = chunk["value"].response
                            break
                    
                    result = {
                        "conv_id": conv_id,
                        "qa_index": i,
                        "question": question,
                        "answer": ground_truth,
                        "hypothesis": hypothesis,
                        "category": category,
                        "timestamp": time.time()
                    }
                    
                    # Global lock for file write isn't needed for .jsonl append, but let's be safe
                    with open(CHECKPOINT_PATH, "a") as f:
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                    
                    results.append(result)
                except Exception as e:
                    print(f"Error in QA {conv_id}-{i}: {e}")
        
        return results

async def main():
    if not os.path.exists(LOCOMO_DATA_PATH):
        print(f"❌ Error: {LOCOMO_DATA_PATH} not found.")
        return

    with open(LOCOMO_DATA_PATH, "r") as f:
        data = json.load(f)
    
    # Initialize checkpoint (Never delete progress!)
    # if os.path.exists(CHECKPOINT_PATH):
    #     os.remove(CHECKPOINT_PATH)
    
    engine = NeuralMemoryEngine()
    
    # Missing Initialization: Connect to Cloud Qdrant and OpenAI
    from qdrant_client import AsyncQdrantClient, QdrantClient
    from openai import AsyncOpenAI
    q_async = AsyncQdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    q_sync = QdrantClient(url=os.getenv("QDRANT_URL"), port=6333, api_key=os.getenv("QDRANT_API_KEY"), https=True)
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    await engine.initialize(q_async, openai_client, sync_qdrant_client=q_sync)
    
    # FINAL SPRINT: 30 concurrent OpenAI calls for 1-minute deadline
    total_semaphore = asyncio.Semaphore(30)
    # Semaphore for conversations
    conv_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    tasks = []
    print(f"🚀 Starting Sequential LoCoMo Evaluation (n={len(data)} conversations)")
    start_time = time.time()
    
    for sample in tqdm(data, desc="Processing Conversations"):
        await process_conversation(sample, engine, conv_semaphore, total_semaphore)
        print(f"😴 Cooling down for {COOLDOWN_SECONDS}s...")
        await asyncio.sleep(COOLDOWN_SECONDS)
    
    duration = time.time() - start_time
    print(f"✅ Evaluation complete in {duration/60:.2f} minutes.")
    print(f"Results recorded in: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
