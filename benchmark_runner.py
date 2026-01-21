import asyncio
import json
import time
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
from neural_memory import NeuralMemoryEngine
from response_generator import react_agent, ResponseGeneratorOutput, sync_client as openai_client
from vectordb import client as qdrant_client, get_all_categories
from embed import generate_embeddings
from rich.console import Console
from rich.progress import Progress

console = Console()

async def retry_with_backoff(func, *args, **kwargs):
    max_retries = 3
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if i == max_retries - 1:
                raise e
            wait_time = 2 * (i + 1)
            console.log(f"[yellow]⚠️ Task failed with {e}. Retrying in {wait_time}s...[/yellow]")
            await asyncio.sleep(wait_time)

async def add_memory_task(engine, user_id, user_handle, text, timestamp):
    """Wait for the memory to be stored with the correct timestamp"""
    async def _store():
        await engine.store_memory(
            user_id=user_id,
            memory_text=text,
            categories=[],
            user_handle=user_handle,
            timestamp=timestamp
        )
    await retry_with_backoff(_store)

async def generate_response_task(engine, user_id, user_handle, question, category_cache):
    # 1. Topic Splitting & Multi-Prefetch (Simulate what run_chat does)
    sub_queries = [
        s.strip()
        for s in question.replace(" and ", "|")
        .replace(" also ", "|")
        .replace("?", "")
        .split("|")
    ]
    if len(sub_queries) == 1:
        # Plus a broad version to ensure high recall
        sub_queries.append(f"identity and personal background of the user")
    
    try:
        emb_res = await generate_embeddings(sub_queries)
        
        search_tasks = [
            engine.get_enhanced_search_memories(
                emb, user_id, user_handle=user_handle, search_text=sub_q
            )
            for emb, sub_q in zip(emb_res, sub_queries)
        ]
        results = await asyncio.gather(*search_tasks)
        
        all_m = []
        seen_ids = set()
        for cluster in results:
            for m in cluster:
                if m.point_id not in seen_ids:
                    all_m.append(m)
                    seen_ids.add(m.point_id)
        
        # 2. Call the agent
        final_response = "No response generated."
        transcript = [{"role": "user", "content": question}] 
        
        async for chunk in react_agent(
            user_id=user_id,
            transcript=transcript,
            question=question,
            existing_categories=category_cache,
            tools={},
            neural_memory_engine=engine,
            max_iters=2,
            user_handle=user_handle,
            prefetched_memories=all_m,
        ):
            if chunk["type"] == "final_output":
                final_response = chunk["value"].response
                
        return final_response
    except Exception as e:
        console.log(f"[bold red]Error in generate_response_task: {e}[/bold red]")
        return f"Error: {e}"

def compute_haystack_hash(item: Dict[str, Any]) -> str:
    """Compute a hash of the haystack to detect identical histories."""
    import hashlib
    haystack_str = json.dumps(item.get('haystack_sessions', []), sort_keys=True)
    return hashlib.md5(haystack_str.encode()).hexdigest()

async def run_benchmark(sample_size: int = 10, use_cache: bool = True):
    """
    Run the LongMemEval benchmark with optimizations.
    
    Args:
        sample_size: Number of questions to evaluate (default: 20 for fast testing)
        use_cache: Whether to cache and reuse haystacks with identical content
    """
    console.print("[bold blue]🚀 Starting Radix-Titan LongMemEval Benchmark[/bold blue]")
    console.print(f"[dim]📊 Sample Size: {sample_size} | Haystack Caching: {'ON' if use_cache else 'OFF'}[/dim]")
    
    # Strategy Guide
    if sample_size <= 50:
        console.print("[green]🧪 Proof of Concept Mode[/green]: Perfect for testing logic (Radix-Titan) without waiting hours.")
    elif 100 <= sample_size <= 150:
        console.print("[yellow]📜 Publication Mode[/yellow]: Statistical significance achieved. Good for final results table.")
    elif sample_size > 400:
        console.print("[red]⚠️  Overkill Mode[/red]: You only need this for the absolute final run. Expect long wait times.")
    
    # Setup
    user_id = 999
    base_user_handle = f"bench_{int(time.time())}"
    
    # Initialize engine
    engine = NeuralMemoryEngine()
    await engine.initialize(qdrant_client, openai_client)
    
    # Load dataset
    data_path = "data/longmemeval_s_cleaned.json"
    if not os.path.exists(data_path):
        console.print(f"❌ Dataset not found at {data_path}. Please ensure it is downloaded.", style="bold red")
        return

    try:
        with open(data_path, "r") as f:
            dataset = json.load(f)
    except Exception as e:
        console.print(f"❌ Failed to load dataset: {e}", style="bold red")
        return

    results = []
    
    # =====================================================
    # OPTIMIZATION 1: Sub-sample the dataset for faster testing
    # =====================================================
    eval_set = dataset[:sample_size] if sample_size > 0 else dataset
    console.print(f"[bold cyan]📋 Evaluating {len(eval_set)} / {len(dataset)} questions[/bold cyan]")
    
    # =====================================================
    # OPTIMIZATION 2: Haystack Caching (Persistent Indexing)
    # Cache user handles by haystack hash to avoid re-ingesting identical histories
    # =====================================================
    haystack_cache: Dict[str, str] = {}  # hash -> user_handle
    category_cache_map: Dict[str, List[str]] = {}  # user_handle -> categories
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Evaluating...", total=len(eval_set))
        
        for i, item in enumerate(eval_set):
            haystack_hash = compute_haystack_hash(item) if use_cache else None
            
            # Check if we already indexed this exact haystack
            if use_cache and haystack_hash in haystack_cache:
                current_user_handle = haystack_cache[haystack_hash]
                categories = category_cache_map.get(current_user_handle, [])
                console.log(f"[dim]⚡ Cache hit for haystack {haystack_hash[:8]}...[/dim]")
            else:
                current_user_handle = f"{base_user_handle}_{i}"
                
                # 1. Ingest History (The "Add" stage)
                sessions = item.get('haystack_sessions', [])
                dates = item.get('haystack_dates', [])
                
                for session, date_str in zip(sessions, dates):
                    # Parse timestamp
                    try:
                        # Format: "2023/05/20 (Sat) 02:21"
                        dt = datetime.strptime(date_str, "%Y/%m/%d (%a) %H:%M")
                        ts = dt.timestamp()
                    except Exception:
                        ts = time.time()  # Fallback
                    
                    # Construct text from session messages
                    session_text = ""
                    for msg in session:
                        role = msg.get('role', 'unknown').capitalize()
                        content = msg.get('content', '')
                        session_text += f"{role}: {content}\n"
                    
                    await add_memory_task(engine, user_id, current_user_handle, session_text.strip(), ts)
                
                # Refresh categories for the agent
                categories = await get_all_categories(user_id, current_user_handle)
                
                # Store in cache for future use
                if use_cache and haystack_hash:
                    haystack_cache[haystack_hash] = current_user_handle
                    category_cache_map[current_user_handle] = categories
            
            # 2. Ask the Question (The "Answer" stage)
            start_time = time.perf_counter()
            response = await generate_response_task(
                engine, user_id, current_user_handle, item['question'], categories
            )
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            
            results.append({
                "question_id": item['question_id'],
                "hypothesis": response,
                "latency": latency
            })
            
            progress.update(task, advance=1, description=f"[cyan]Evaluating {item['question_id']} ({latency:.2f}s)")

    # Save for the LongMemEval scorer (NeurIPS-style jsonl)
    output_file = "my_results.jsonl"
    with open(output_file, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
            
    console.print(f"\n[bold green]✅ Benchmark complete![/bold green]")
    console.print(f"📊 Results saved to: [bold]{output_file}[/bold]")
    
    # Calculate average latency
    avg_latency = sum(r['latency'] for r in results) / len(results) if results else 0
    console.print(f"⏱️ Average Latency: [bold]{avg_latency:.2f}s[/bold]")
    
    # Show cache stats
    if use_cache:
        cache_hits = len(eval_set) - len(haystack_cache)
        console.print(f"🔄 Cache Stats: {len(haystack_cache)} unique haystacks, {cache_hits} cache hits")
    
    console.print("\n[bold yellow]Next Steps:[/bold yellow]")
    console.print("1. Run the official LongMemEval scorer on 'my_results.jsonl'.")
    console.print("2. Report Accuracy (Acc) and Semantic Integrity (Conflict Awareness).")


async def run_benchmark_parallel(sample_size: int = 10, num_workers: int = 4):
    """
    Run benchmark with parallel question evaluation (for M2/M3 chips).
    
    Note: This parallelizes the QUESTION evaluation, not the haystack ingestion.
    Haystack ingestion must remain sequential for causality.
    """
    console.print("[bold blue]🚀 Starting PARALLEL Radix-Titan LongMemEval Benchmark[/bold blue]")
    console.print(f"[dim]📊 Sample Size: {sample_size} | Workers: {num_workers}[/dim]")

    # Strategy Guide
    if sample_size <= 50:
        console.print("[green]🧪 Proof of Concept Mode[/green]: Perfect for testing logic (Radix-Titan) without waiting hours.")
    elif 100 <= sample_size <= 150:
        console.print("[yellow]📜 Publication Mode[/yellow]: Statistical significance achieved. Good for final results table.")
    elif sample_size > 400:
        console.print("[red]⚠️  Overkill Mode[/red]: You only need this for the absolute final run. Expect long wait times.")
    
    # Setup
    user_id = 999
    base_user_handle = f"bench_{int(time.time())}"
    
    # Initialize engine
    engine = NeuralMemoryEngine()
    await engine.initialize(qdrant_client, openai_client)
    
    # Load dataset
    data_path = "data/longmemeval_s_cleaned.json"
    if not os.path.exists(data_path):
        console.print(f"❌ Dataset not found at {data_path}.", style="bold red")
        return

    with open(data_path, "r") as f:
        dataset = json.load(f)
    
    eval_set = dataset[:sample_size] if sample_size > 0 else dataset
    console.print(f"[bold cyan]📋 Evaluating {len(eval_set)} / {len(dataset)} questions in parallel[/bold cyan]")
    
    # First, ingest ALL haystacks (sequential for causality)
    console.print("[yellow]📥 Phase 1: Ingesting haystacks...[/yellow]")
    handles_and_cats = []
    
    haystack_cache: Dict[str, tuple] = {}
    
    for i, item in enumerate(eval_set):
        haystack_hash = compute_haystack_hash(item)
        
        if haystack_hash in haystack_cache:
            handles_and_cats.append(haystack_cache[haystack_hash])
            continue
            
        current_user_handle = f"{base_user_handle}_{i}"
        sessions = item.get('haystack_sessions', [])
        dates = item.get('haystack_dates', [])
        
        for session, date_str in zip(sessions, dates):
            try:
                dt = datetime.strptime(date_str, "%Y/%m/%d (%a) %H:%M")
                ts = dt.timestamp()
            except Exception:
                ts = time.time()
            
            session_text = "\n".join(
                f"{msg.get('role', '').capitalize()}: {msg.get('content', '')}"
                for msg in session
            )
            await add_memory_task(engine, user_id, current_user_handle, session_text.strip(), ts)
        
        categories = await get_all_categories(user_id, current_user_handle)
        haystack_cache[haystack_hash] = (current_user_handle, categories)
        handles_and_cats.append((current_user_handle, categories))
    
    # Now, evaluate questions in parallel batches
    console.print(f"[yellow]⚡ Phase 2: Evaluating {len(eval_set)} questions ({num_workers} at a time)...[/yellow]")
    
    results = []
    
    async def eval_single(idx: int, item: Dict, handle: str, cats: List[str]):
        start = time.perf_counter()
        resp = await generate_response_task(engine, user_id, handle, item['question'], cats)
        latency = time.perf_counter() - start
        return {
            "question_id": item['question_id'],
            "hypothesis": resp,
            "latency": latency
        }
    
    # Process in batches of num_workers
    for batch_start in range(0, len(eval_set), num_workers):
        batch_end = min(batch_start + num_workers, len(eval_set))
        batch = eval_set[batch_start:batch_end]
        batch_handles = handles_and_cats[batch_start:batch_end]
        
        tasks = [
            eval_single(batch_start + j, item, h, c)
            for j, (item, (h, c)) in enumerate(zip(batch, batch_handles))
        ]
        
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        console.print(f"[dim]✓ Completed batch {batch_start+1}-{batch_end}[/dim]")
    
    # Save results
    output_file = "my_results.jsonl"
    with open(output_file, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    
    console.print(f"\n[bold green]✅ Parallel Benchmark complete![/bold green]")
    console.print(f"📊 Results saved to: [bold]{output_file}[/bold]")
    
    avg_latency = sum(r['latency'] for r in results) / len(results) if results else 0
    console.print(f"⏱️ Average Latency: [bold]{avg_latency:.2f}s[/bold]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark with optimizations")
    parser.add_argument("-n", "--sample-size", type=int, default=10,
                        help="Number of questions to evaluate (default: 10, use 0 for all)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable haystack caching (rebuild radix tree every time)")
    parser.add_argument("-p", "--parallel", action="store_true",
                        help="Use parallel evaluation (recommended for M2/M3 chips)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    if args.parallel:
        asyncio.run(run_benchmark_parallel(
            sample_size=args.sample_size,
            num_workers=args.workers
        ))
    else:
        asyncio.run(run_benchmark(
            sample_size=args.sample_size,
            use_cache=not args.no_cache
        ))
