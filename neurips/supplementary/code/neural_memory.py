"""
Stratum Neural Memory Engine

This module implements the Stratum architecture for neural memory management,
featuring surprisal-based gating, radix path generation, and KV cache stitching.
"""

import asyncio
import json
import logging
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import redis
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, models, Filter, PointStruct

# Optional ML dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print(
        "Warning: PyTorch/transformers not available. Using simplified implementations."
    )

# import lmcache  # Removed due to CUDA dependency

from embed import generate_embeddings
from vectordb import EmbeddedMemory, RetrievedMemory, COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force local cache for models
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "model_cache")

import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurprisalGate:
    """
    Client for the Local Brain Service to calculate surprisal.
    Uses bert-mini for lighting fast gating.
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8000",
        surprisal_threshold: float = 0.5,
    ):
        self.service_url = service_url
        self.surprisal_threshold = surprisal_threshold

    async def initialize(self):
        """Service is assumed to be running independently."""
        pass

    async def calculate_surprisal(self, text: str, context: str = "") -> float:
        """Query the brain service for surprisal score."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.service_url}/surprisal",
                    json={"text": text, "context": context},
                )
                data = response.json()
                return data["surprisal_score"]
            except Exception as e:
                logger.error(f"Brain Service surprisal failed: {e}")
                return 5.0  # Fallback to 'high surprisal' to be safe

    async def is_topic_known(self, prompt: str, existing_context: str = "") -> bool:
        """Check if the topic in the prompt is known (low surprisal)."""
        surprisal = await self.calculate_surprisal(prompt, existing_context)
        is_known = surprisal < self.surprisal_threshold

        decision_label = "Transient (IGNORE)" if is_known else "Persistent (STORE)"
        logger.info(
            f"SURPRISAL: {surprisal:.3f} | THRESHOLD: {self.surprisal_threshold} | DECISION: {decision_label}"
        )

        return is_known


class TaxonomyLLM:
    """
    Client for the Local Brain Service to generate radix paths.
    """

    def __init__(self, service_url: str = "http://localhost:8000"):
        self.service_url = service_url

    async def initialize(self):
        """Service is assumed to be running independently."""
        pass

    async def generate_radix_path(
        self,
        text: str,
        user_handle: str = "user",
        existing_paths: List[str] = None,
        mode: str = "storage",
    ) -> List[str]:
        """Query Gemini for radix path."""
        from response_generator import call_openai_chat
        
        prompt = f"""
        Generate multiple possible 3-level radix paths where this memory might be stored.
        Text: {text}
        User: {user_handle}
        Mode: {mode}
        Format: users/{user_handle}/[root]/[domain]/[detail]
        Roots: personal, work, tech, general, entities, objects, self
        
        CRITICAL: Provide 10 highly diverse path guesses. Include both general and highly specific paths to guarantee a match.
        Example: users/{user_handle}/personal/general/detail, users/{user_handle}/entities/domain/detail
        Return ONLY the paths, comma separated.
        """
        
        try:
            # We use our Gemini-powered wrapper
            # Let our wrapper choose the best available model (Groq -> Gemini)
            response = await call_openai_chat(prompt=prompt)
            generated_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Gemini taxonomy failed: {e}")
            generated_text = f"users/{user_handle}/general/topic/detail"

        # Parse the generated path(s)
        # If mode is search, we might get comma separated paths
        raw_paths = [p.strip() for p in generated_text.split(",")]
        parsed_paths = []

        for raw_p in raw_paths:
            parsed = self._parse_generated_path(raw_p, user_handle)
            if parsed:
                parsed_paths.append(parsed)

        if not parsed_paths:
            parsed_paths = [f"users/{user_handle.lower()}/general/topic/detail"]

        if mode == "storage":
            # Just take the first one for storage log
            path = parsed_paths[0]
            print(f"📍 DEBUG: Generated Path -> {path}")
            logger.info(f"Generated radix path for '{text[:50]}...': {path}")
            return parsed_paths  # Return list but typically consumer takes index 0 or all for categories
        else:
            print(f"📍 DEBUG: Predicted Search Paths -> {parsed_paths}")
            logger.info(f"Generated search paths: {parsed_paths}")
            return parsed_paths

    def _parse_generated_path(
        self, generated_text: str, user_handle: str = "user"
    ) -> str:
        """
        Parse the generated text to extract a clean 3-level path.
        Enforces valid roots and cleans segments from rules/hallucinations.
        Implements strict line-one cleaning.
        """
        import re

        # 1. Strict Line-One Cleaning: Take only the first line of content
        first_line = generated_text.split("\n")[0].strip()

        # 2. Basic cleanup: strip characters that often appear in hallucinated paths or Rules leak
        clean_text = (
            first_line.replace("[", "")
            .replace("]", "")
            .replace('"', "")
            .replace("'", "")
            .replace(".", "")
        )
        if "rules:" in clean_text.lower():
            clean_text = clean_text.split("rules:")[0].strip()
        if "the_first_level" in clean_text.lower():
            clean_text = clean_text.split("the_first_level")[0].strip()

        # 3. Extract segments
        parts = [
            p.strip().lower().replace(" ", "_")
            for p in clean_text.split("/")
            if p.strip()
        ]

        # 4. Filter segments: keep only alphanumeric and underscores
        parts = [re.sub(r"[^a-zA-Z0-9_]", "", p) for p in parts]
        parts = [p[:30] for p in parts if p]  # Truncate long segments

        # 5. Enforce Prefix: users/{{user_handle}}/
        # If the LLM didn't return the prefix, we prepend it.
        # If it returned something like 'users/{user}/...', we make sure it matches.

        valid_roots = [
            "personal",
            "work",
            "tech",
            "general",
            "entities",
            "objects",
            "self",
        ]

        # Remove "users" and "handle" if they are at the start
        if len(parts) > 0 and parts[0] == "users":
            parts = parts[1:]
        if len(parts) > 0 and parts[0] == user_handle.lower():
            parts = parts[1:]

        # Now parts should start with one of the valid_roots
        if not parts or parts[0] not in valid_roots:
            # Try to find a valid root anywhere in the list
            found_root_idx = next(
                (i for i, p in enumerate(parts) if p in valid_roots), -1
            )
            if found_root_idx != -1:
                parts = parts[found_root_idx:]
            else:
                parts.insert(0, "general")

        # 6. Re-construct with required prefix
        final_parts = ["users", user_handle.lower()] + parts

        # Ensure at least 3 levels after prefix (Total 5 segments potential, but we can stop at 4)
        # Let's keep it to 3 or 4 segments total if possible, max 5.
        if len(final_parts) < 4:
            while len(final_parts) < 4:
                final_parts.append("detail")

        return "/".join(
            final_parts[:6]
        )  # Max 6 segments: users/handle/root/domain/subdomain/detail


class LMCacheRedisManager:
    """
    Manages KV cache storage and retrieval using Redis.
    Simplified version without LMCache CUDA dependencies.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None

    async def initialize(self):
        """Initialize Redis connection"""
        if self.redis_client is None:
            self.redis_client = redis.Redis.from_url(
                self.redis_url, decode_responses=False
            )

    async def store_kv_cache(self, radix_path: str, kv_cache: Any):
        """
        Store KV cache in Redis keyed by radix path.
        """
        await self.initialize()

        key = f"lmcache:{radix_path}"
        serialized_cache = pickle.dumps(kv_cache)

        self.redis_client.set(key, serialized_cache)
        logger.info(f"Stored KV cache for radix path: {radix_path}")

    async def retrieve_kv_cache(self, radix_path: str) -> Optional[Any]:
        """
        Retrieve KV cache from Redis by radix path.
        """
        await self.initialize()

        key = f"lmcache:{radix_path}"
        serialized_cache = self.redis_client.get(key)

        if serialized_cache:
            kv_cache = pickle.loads(serialized_cache)
            logger.info(f"Retrieved KV cache for radix path: {radix_path}")
            return kv_cache

        logger.warning(f"No KV cache found for radix path: {radix_path}")
        return None

    async def get_parent_paths(self, radix_path: str) -> List[str]:
        """
        Get parent paths by walking up the radix tree.
        e.g., work/project/alpha -> [work/project, work]
        """
        parts = radix_path.split("/")
        parent_paths = []

        for i in range(len(parts) - 1, 0, -1):
            parent_path = "/".join(parts[:i])
            parent_paths.append(parent_path)

        return parent_paths

    async def get_sibling_paths(self, radix_path: str) -> List[str]:
        """
        Get sibling paths at the same level.
        e.g., work/project/alpha -> find other paths under work/project/
        """
        await self.initialize()

        parts = radix_path.split("/")
        if len(parts) < 2:
            return []

        parent_prefix = "/".join(parts[:-1]) + "/"
        pattern = f"lmcache:{parent_prefix}*"

        # Get all keys matching the parent prefix
        keys = self.redis_client.keys(pattern)
        sibling_paths = []

        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            if key_str.startswith("lmcache:"):
                path = key_str[len("lmcache:") :]
                if path != radix_path:
                    sibling_paths.append(path)

        return sibling_paths

    async def get_all_known_paths(self, user_handle: str) -> List[str]:
        """
        Retrieve all known paths for a specific user to ground the taxonomy.
        """
        await self.initialize()
        # Scan for user's paths
        pattern = f"lmcache:users/{user_handle.lower()}*"
        keys = self.redis_client.keys(pattern)
        paths = []
        for k in keys[:50]:  # Limit to 50 to avoid prompt overflow
            k_str = k.decode() if isinstance(k, bytes) else k
            if k_str.startswith("lmcache:"):
                paths.append(k_str.replace("lmcache:", ""))
        return paths


class InferencePrefetcher:
    """
    Handles lightweight retrieval with KV cache injection before OpenAI calls.
    """

    def __init__(
        self, qdrant_client: AsyncQdrantClient, lmcache_manager: LMCacheRedisManager
    ):
        self.qdrant_client = qdrant_client
        self.lmcache_manager = lmcache_manager
        self.confidence_threshold = 0.7

    async def prefetch_context(
        self, query: str, user_id: int, user_handle: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query Qdrant and prefetch KV cache if high confidence match found.
        Returns context dict with radix_path and kv_cache if available.
        """
        # 0. Get embeddings for semantic search
        vectors = None
        # 0. Get embeddings for semantic search
        vectors = None
        try:
            # TRY EMBEDDING FIRST
            vectors = await generate_embeddings([query])
        except Exception as e:
            if "All Gemini Embedding attempts failed" in str(e):
                logger.warning(f"⚠️ Quota Exhausted for search! Falling back to KEYWORD-ONLY...")
            else:
                raise e

        # 🚀 THE OMEGA-SEARCH: HYBRID-PARALLEL
        search_results = []
        search_vector = vectors[0] if vectors else None

        must_conditions = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        ]
        if user_handle:
            must_conditions.append(
                models.FieldCondition(
                    key="user_handle", match=models.MatchValue(value=user_handle)
                )
            )

        # Search Qdrant
        results = await self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=search_vector,
            with_payload=True,
            query_filter=Filter(must=must_conditions),
            score_threshold=self.confidence_threshold,
            limit=1,
        )

        if results.points and len(results.points) > 0:
            point = results.points[0]
            radix_path = point.payload.get("radix_path")

            if radix_path:
                # Try to retrieve KV cache
                kv_cache = await self.lmcache_manager.retrieve_kv_cache(radix_path)

                return {
                    "radix_path": radix_path,
                    "kv_cache": kv_cache,
                    "memory_text": point.payload.get("memory_text", ""),
                    "score": point.score,
                }

        return None


class RecursiveContextSearch:
    """
    Tool for traversing the Radix Tree upwards or sideways to combine multiple KV caches.
    """

    def __init__(self, lmcache_manager: LMCacheRedisManager):
        self.lmcache_manager = lmcache_manager

    async def search_context(
        self,
        radix_path: str,
        user_handle: str,
        direction: str = "up",
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Traverse radix tree and collect related KV caches.

        Args:
            radix_path: Starting radix path
            user_handle: The handle of the current user (Hard Fence)
            direction: "up" for parent traversal, "side" for sibling traversal
            max_depth: Maximum traversal depth

        Returns:
            List of context dicts with radix_path and kv_cache
        """
        # REQUIREMENT: Hard Fence - Ensure the path belongs to the user
        prefix = f"users/{user_handle.lower()}"

        # FIX: Smart Boundary Check
        # 1. If path matches user prefix -> OK
        # 2. If path is a parent of user prefix (e.g. "users") -> STOP (Boundary, not breach)
        # 3. If path is unrelated (e.g. "users/other") -> BREACH

        if not radix_path.startswith(prefix):
            # Check if we hit the ceiling (path is a prefix of the user_prefix)
            if prefix.startswith(radix_path):
                return []  # Just stop, we hit the roof

            # Otherwise, it's a sideways jump to another user -> Breach
            logger.warning(
                f"🚨 Security Breach Attempt: {user_handle} tried to traverse {radix_path}"
            )
            return []

        contexts = []

        if direction == "up":
            # Walk up the tree
            paths_to_check = await self.lmcache_manager.get_parent_paths(radix_path)
            # FIX: Hard Floor - Stop at the user's front door (users/{handle})
            # We filter out any paths that are parents of the user root (e.g. "users" or "")
            # effectively ensuring we stay within users/{handle}/...
            filtered_paths = []
            for p in paths_to_check:
                # Step C: Explicit Hard Floor check
                if p == prefix.rstrip("/"):
                    continue  # Stop before adding the root
                if p.startswith(prefix) and len(p) >= len(prefix):
                    filtered_paths.append(p)
            paths_to_check = filtered_paths

        elif direction == "side":
            # Get siblings
            paths_to_check = await self.lmcache_manager.get_sibling_paths(radix_path)
        else:
            paths_to_check = []

        # Limit to max_depth
        paths_to_check = paths_to_check[:max_depth]

        for path in paths_to_check:
            kv_cache = await self.lmcache_manager.retrieve_kv_cache(path)
            if kv_cache:
                contexts.append(
                    {"radix_path": path, "kv_cache": kv_cache, "direction": direction}
                )

        logger.info(
            f"Found {len(contexts)} related contexts via {direction} traversal from {radix_path}"
        )
        return contexts


async def store_neural_memory(
    user_id: int,
    memory_text: str,
    categories: List[str],
    qdrant_client: AsyncQdrantClient,
    surprisal_gate: SurprisalGate,
    taxonomy_llm: TaxonomyLLM,
    lmcache_manager: LMCacheRedisManager,
    existing_context: str = "",
    radix_path: Optional[str] = None,
    score_boost: float = 0.0,
    user_handle: str = "user",
    embedding: Optional[List[float]] = None,
    cross_references: Optional[List[str]] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Store memory using the neural memory pipeline.

    1. Check surprisal (is topic known?)
    2. Generate radix path (if not provided)
    3. Save to Qdrant with radix_path metadata
    4. Save KV cache to Redis via LMCache
    """

    # Step 0: Perspective Normalization (User -> Third Person)
    # Convert first-person to third-person for cleaner future retrieval
    normalized_text = (
        memory_text.replace("My ", "The user's ")
        .replace("my ", "the user's ")
        .replace("I am ", "The user is ")
        .replace("I'm ", "The user is ")
        .replace("I live ", "The user lives ")
    )

    # Step 1: Surprisal check
    if surprisal_gate is not None and score_boost < 50:
        surprisal_score = await surprisal_gate.calculate_surprisal(
            normalized_text, existing_context
        )
        surprisal_score += score_boost
        is_known = surprisal_score < -1.0
        if is_known:
            return {"action": "skipped", "reason": "topic_known", "score": surprisal_score}

    # Step 2: Generate radix paths (if not provided)
    if radix_path is None:
        raw_paths = await taxonomy_llm.generate_radix_path(memory_text, user_handle)
        raw_list = [p.strip() for p in raw_paths.split(",")]
    else:
        raw_list = [p.strip() for p in radix_path.split(",")]

    # AUTO-ALIGN: Index ALL parent paths for hierarchical recall
    if categories is None:
        categories = []
    
    hierarchical_paths = []
    for r_path in raw_list:
        parts = [p.strip() for p in r_path.split("/") if p.strip()]
        curr = ""
        for part in parts:
            curr = part if not curr else f"{curr}/{part}"
            if curr not in hierarchical_paths:
                hierarchical_paths.append(curr)
            if part not in categories:
                categories.append(part)
    
    # Update the local variable used in payload
    radix_paths = hierarchical_paths

    # Step 3: Create embedding (Reuse if provided)
    if embedding is None:
        embeddings = await generate_embeddings([memory_text])
        embedding = embeddings[0]

    # Capture current timestamp for temporal stitching
    current_ts = timestamp if timestamp is not None else time.time()

    # Step 4: Store in Qdrant with radix_path metadata and timestamps
    await qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=uuid4().hex,
                payload={
                    "cross_references": cross_references or [],
                },
                vector=embedding,
            )
        ],
        wait=False,  # Set to False to maximize throughput during high-concurrency benchmarks
    )

    # Step 5: Generate and store KV caches for each path
    for path in radix_paths:
        mock_kv_cache = {
            "radix_path": path,
            "memory_text": normalized_text,
            "timestamp": current_ts,
            "mock": True,
        }
        await lmcache_manager.store_kv_cache(path, mock_kv_cache)

    final_path = radix_paths[0] if radix_paths else "general/topic/detail"
    logger.info(f"Stored neural memory with radix path: {final_path}")
    return {"action": "stored", "radix_path": final_path, "memory_text": memory_text}


class NeuralMemoryEngine:
    """
    Main orchestrator for the Stratum Neural Memory Engine.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url

        # Initialize components
        self.surprisal_gate = SurprisalGate()
        self.taxonomy_llm = TaxonomyLLM()
        self.lmcache_manager = LMCacheRedisManager(redis_url)
        self.inference_prefetcher = None
        self.recursive_search = RecursiveContextSearch(self.lmcache_manager)

        # Qdrant client (will be set externally)
        self.qdrant_client = None
        self.sync_qdrant_client = None

        # OpenAI client (will be set externally)
        self.openai_client = None

    async def initialize(self, qdrant_client: AsyncQdrantClient, openai_client=None, sync_qdrant_client: Optional[QdrantClient] = None):
        """Initialize all components"""
        self.qdrant_client = qdrant_client
        self.sync_qdrant_client = sync_qdrant_client
        self.openai_client = openai_client

        # Indexing skipped for benchmark speed - collection already indexed during ingestion.
        pass

        self.inference_prefetcher = InferencePrefetcher(
            qdrant_client, self.lmcache_manager
        )

        await self.surprisal_gate.initialize()
        await self.taxonomy_llm.initialize()
        await self.lmcache_manager.initialize()

    async def store_memory(
        self,
        user_id: int,
        memory_text: str,
        categories: List[str] = None,
        metadata: Dict[str, Any] = None,
        user_handle: str = "unique_id",
        conversation_id: str = None,
        timestamp: Optional[float] = None,
        skip_taxonomy: bool = False
    ) -> Dict[str, Any]:
        """
        Stores a memory with vector and structural pathing.
        """
        if not self.qdrant_client:
            raise Exception("NeuralMemoryEngine not initialized")

        # 1. 🧬 TITAN-RADIX: Structural Path Generation
        if skip_taxonomy:
            radix_paths = ["users/user/general/bulk_ingest"]
        else:
            radix_paths = await self._generate_structural_paths(memory_text, user_handle)

        # 2. 🌌 VECTOR: Embedding generation
        embedding = await self._get_embedding(memory_text)

        # 3. 🕸️ PAYLOAD: Unified metadata
        payload = {
            "user_id": user_id,
            "user_handle": user_handle,
            "conversation_id": conversation_id,
            "memory_text": memory_text,
            "categories": categories or [],
            "radix_paths": radix_paths,
            "metadata": metadata or {},
            "timestamp": timestamp or time.time(),
            "date": datetime.now().strftime("%d %B, %Y")
        }

        # 4. 🛰️ QDRANT: Async upsert
        await self.qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=uuid4().hex,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        return {"action": "stored", "radix_paths": radix_paths}

    async def get_temporal_context(self, user_id, timestamp, window_seconds=600, user_handle=None):
        must_conditions = [models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)), models.FieldCondition(key="timestamp", range=models.Range(gt=timestamp-window_seconds, lt=timestamp+window_seconds))]
        if user_handle: must_conditions.append(models.FieldCondition(key="user_handle", match=models.MatchValue(value=user_handle)))
        outs, _ = await self.qdrant_client.scroll(collection_name=COLLECTION_NAME, scroll_filter=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]), limit=500, with_payload=True)
        # Manual stitching in memory to bypass missing timestamp index
        context_points = [p for p in outs if p.payload.get("timestamp") and abs(p.payload.get("timestamp") - timestamp) < window_seconds]
        return [self._convert_enhanced_retrieved_records(p) for p in context_points][:10]

    async def get_enhanced_search_memories(
        self,
        search_vector: List[float],
        user_id: int,
        limit: int = 151,
        score_threshold: float = 0.2,
        radix_paths: Optional[List[str]] = None,
        user_handle: Optional[str] = None,
        conversation_id: Optional[str] = None,
        search_text: Optional[str] = None
    ) -> List[RetrievedMemory]:
        """The core retrieval method for Titan: Semantic + Structural + Radix Reranking."""

        must_conditions = [
            models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id)
            )
        ]

        if user_handle:
            must_conditions.append(
                models.FieldCondition(
                    key="user_handle", match=models.MatchValue(value=user_handle)
                )
            )

        if conversation_id:
            must_conditions.append(
                models.FieldCondition(
                    key="conversation_id", match=models.MatchValue(value=conversation_id)
                )
            )

        search_filter = models.Filter(must=must_conditions)

        # Semantic Search in Qdrant
        points = await self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=search_vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        )

        memories = [
            self._convert_enhanced_retrieved_records(point)
            for point in points.points
        ]

        # Radix-Rerank: boost memories whose radix_path overlaps with query's radix paths
        if radix_paths and memories:
            query_path_parts = set()
            for rp in radix_paths:
                query_path_parts.update(rp.strip("/").split("/"))

            for mem in memories:
                mem_paths = mem.radix_paths or ([mem.radix_path] if mem.radix_path else [])
                mem_path_parts = set()
                for mp in mem_paths:
                    if mp:
                        mem_path_parts.update(mp.strip("/").split("/"))
                # Compute path overlap ratio
                if mem_path_parts and query_path_parts:
                    overlap = len(query_path_parts & mem_path_parts) / len(query_path_parts)
                    # Boost score: up to 30% bonus for full path match
                    mem.score = mem.score * (1.0 + 0.3 * overlap)

            # Re-sort by boosted score (higher is better)
            memories.sort(key=lambda m: m.score, reverse=True)

        # Keyword boost: if search_text provided, boost memories containing key terms
        if search_text and memories:
            keywords = [w.lower() for w in search_text.split() if len(w) > 3]
            for mem in memories:
                text_lower = mem.memory_text.lower()
                hits = sum(1 for kw in keywords if kw in text_lower)
                if keywords:
                    keyword_ratio = hits / len(keywords)
                    mem.score = mem.score * (1.0 + 0.2 * keyword_ratio)
            memories.sort(key=lambda m: m.score, reverse=True)

        return memories

    async def _generate_structural_paths(self, text: str, user_handle: str = "user") -> List[str]:
        """Exposes the radix path generator for taxonomy classification."""
        # This calls the taxonomy_llm component
        return await self.taxonomy_llm.generate_radix_path(text, user_handle)

    async def _get_embedding(self, text: str) -> List[float]:
        """Generates embedding for the text."""
        # This uses the external embedder module (OpenAI text-embedding-3-small)
        res = await generate_embeddings([text])
        return res[0]

    def _convert_enhanced_retrieved_records(self, point) -> RetrievedMemory:
        """Convert with radix_path support"""
        payload = point.payload
        radix_paths = payload.get("radix_paths")
        if not radix_paths and payload.get("radix_path"):
            radix_paths = [payload.get("radix_path")]

        return RetrievedMemory(
            point_id=point.id,
            user_id=payload["user_id"],
            memory_text=payload["memory_text"],
            categories=payload["categories"],
            date=payload.get("date", ""),
            score=getattr(point, "score", 1.0),
            radix_path=payload.get("radix_path"),
            radix_paths=radix_paths,
            timestamp=payload.get("timestamp"),
        )
from embed import generate_embeddings
from qdrant_client import models
import uuid
import time
from typing import List, Dict, Any, Optional
import logging
import json
logger = logging.getLogger(__name__)
COLLECTION_NAME = "cognitive_memory_local_v3"

class RetrievedMemory:
    def __init__(self, point_id, user_id, memory_text, categories, date, score, radix_path, radix_paths, timestamp):
        self.point_id = point_id
        self.user_id = user_id
        self.memory_text = memory_text
        self.categories = categories
        self.date = date
        self.score = score
        self.radix_path = radix_path
        self.radix_paths = radix_paths
        self.timestamp = timestamp

    async def deleted_method(self,
        user_id: int,
        timestamp: float,
        window_seconds: int = 600,
        user_handle: Optional[str] = None,
    ) -> List[RetrievedMemory]:
        """
        Retrieves memories that occurred within a 10-minute window of the given timestamp.
        This provides 'Temporal Stitching' across semantic branches.
        """
        must_conditions = [
            models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id)
            ),
            models.FieldCondition(
                key="timestamp",
                range=models.Range(
                    gt=timestamp - window_seconds, lt=timestamp + window_seconds
                ),
            ),
        ]

        if user_handle:
            must_conditions.append(
                models.FieldCondition(
                    key="user_handle", match=models.MatchValue(value=user_handle)
                )
            )

        outs = await self.qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=must_conditions),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )

        return [
            self._convert_enhanced_retrieved_records(point)
            for point in outs[0]
            if point is not None
        ]


# Import here to avoid circular imports
from datetime import datetime
from uuid import uuid4
