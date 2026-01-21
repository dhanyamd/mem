"""
Radix-Titan Neural Memory Engine

This module implements the Radix-Titan architecture for neural memory management,
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
from qdrant_client import AsyncQdrantClient
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
        surprisal_threshold: float = 2.5,
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
        """Query the brain service for radix path."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.service_url}/taxonomy",
                    json={
                        "text": text,
                        "user_handle": user_handle,
                        "existing_paths": existing_paths or [],
                        "mode": mode,
                    },
                )
                data = response.json()
                generated_text = data["radix_path"]
            except Exception as e:
                logger.error(f"Brain Service taxonomy failed: {e}")
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
        # If it returned something like 'users/dhanya/...', we make sure it matches.

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
        # Generate embedding for the query
        embeddings = await generate_embeddings([query])
        search_vector = embeddings[0]

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
                    "memory_text": point.payload.get("memory_texts", ""),
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
    surprisal_score = await surprisal_gate.calculate_surprisal(
        normalized_text, existing_context
    )
    surprisal_score += score_boost

    is_known = surprisal_score < surprisal_gate.surprisal_threshold

    decision = "IGNORE" if is_known else "STORE"
    logger.info(
        f"MEM_DECISION: {decision} based on score {surprisal_score:.3f} (threshold: {surprisal_gate.surprisal_threshold})"
    )

    if is_known and score_boost < 50:  # Only skip if not forced
        return {"action": "skipped", "reason": "topic_known", "score": surprisal_score}

    # Step 2: Generate radix paths (if not provided)
    # Support for comma-separated multiple paths
    if radix_path is None:
        raw_paths = await taxonomy_llm.generate_radix_path(memory_text, user_handle)
        radix_paths = [p.strip() for p in raw_paths.split(",")]
    else:
        radix_paths = [p.strip() for p in radix_path.split(",")]

    # AUTO-ALIGN: Index ALL parts of the radix path for broad recall
    if not categories:
        categories = []
        for path in radix_paths:
            parts = [p.strip() for p in path.split("/") if p.strip()]
            for part in parts:
                if part not in categories:
                    categories.append(part)

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
                    "user_id": user_id,
                    "user_handle": user_handle,  # REQUIREMENT 3: Hard Metadata Filter
                    "categories": categories,
                    "memory_texts": normalized_text,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "radix_paths": radix_paths,  # Storing list of paths
                    "radix_path": radix_paths[0],  # Legacy support
                    "timestamp": current_ts,  # High-precision for temporal stitching
                    "cross_references": cross_references or [],
                },
                vector=embedding,
            )
        ],
        wait=True,  # Ensuring persistence for safety
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
    Main orchestrator for the Radix-Titan Neural Memory Engine.
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

        # OpenAI client (will be set externally)
        self.openai_client = None

    async def initialize(self, qdrant_client: AsyncQdrantClient, openai_client=None):
        """Initialize all components"""
        self.qdrant_client = qdrant_client
        self.openai_client = openai_client

        # FIX: Ensure payload indexing for radix_paths to prevent 400 errors during summarization
        # We use TEXT index to allow for partial matching in "Smart Filter"
        try:
            await self.qdrant_client.delete_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="radix_paths",
            )
        except Exception:
            pass  # Index might not exist

        try:
            await self.qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="radix_paths",
                field_schema=models.PayloadSchemaType.TEXT,
            )
        except Exception as e:
            # Index might already exist, which is fine
            logger.info(f"Index creation note: {e}")

        self.inference_prefetcher = InferencePrefetcher(
            qdrant_client, self.lmcache_manager
        )

        await self.surprisal_gate.initialize()
        await self.taxonomy_llm.initialize()
        await self.lmcache_manager.initialize()

        logger.info("NeuralMemoryEngine initialized")

    async def detect_conflict(
        self, text: str, memories: List[RetrievedMemory]
    ) -> Optional[Dict[str, Any]]:
        """
        Detects if the new text contradicts any existing memories.
        Returns a dict with conflict info if found, else None.
        """
        if not memories or not self.openai_client:
            return None

        memory_context = "\n".join(
            [f"- [{m.point_id}] {m.memory_text}" for m in memories[:5]]
        )

        prompt = f"""Analyze the new input against the stored memories. 
        Does the new input contradict or update a specific fact in the memories? 
        
        Distinguish between:
        1. "UPDATE": A factual correction (e.g., "I moved to NY" vs "I live in LA"). This requires deleting the old fact.
        2. "WARNING": A contextual clash or safety warning (e.g., "Arjun hates peanuts" vs "Riya loves peanuts"). This does NOT require deletion, but flags a potential issue.
        
        Stored Memories:
        {memory_context}
        
        New Input:
        {text}
        
        Return JSON: {{"conflict": true, "type": "UPDATE" | "WARNING", "conflicting_point_id": "ID", "reason": "why"}}
        If no conflict, return JSON: {{"conflict": false}}.
        Only return valid JSON."""

        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a specialized conflict detection module for a neural memory engine.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            if data.get("conflict") and data.get("conflicting_point_id"):
                return data
        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")

        return None

    async def classify_intent(self, text: str) -> str:
        """
        Ask the LLM (OpenAI) if the input contains a "Personal Fact" (Identity, Preference, Work, Location).
        Returns "Personal Fact", "Small Talk", or "Other".
        """
        if not self.openai_client:
            return "Other"

        prompt = f"""Analyze the user input and classify its intent for memory storage.
- "Personal Fact": Contains specific identity, preference, work, location, or personal details.
- "Memory Deletion": A request to forget or delete previously stored information (e.g., "Forget my name", "Delete what you know about Kerala", "Clear my location").
- "Small Talk": General conversational filler or transient comments.
- "Other": Everything else.

Input: {text}

Return JSON: {{"intent": "Personal Fact" | "Memory Deletion" | "Small Talk" | "Other"}}
Only return valid JSON."""

        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a specialized intent classifier for a neural memory engine.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("intent", "Other")
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "Other"

    async def store_memory(
        self,
        user_id: int,
        memory_text: str,
        categories: List[str],
        existing_context: str = "",
        radix_path: Optional[str] = None,
        user_handle: str = "unique_id",
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Store memory using the complete neural pipeline.
        Includes conflict detection, intent classification, and dynamic thresholding.
        """
        # Step 1-3: Parallel Processing (SPEED-PRO)
        # We run Embedding, Intent, and Surprisal in parallel
        tasks = [
            generate_embeddings([memory_text]),
            self.classify_intent(memory_text),
            self.surprisal_gate.calculate_surprisal(memory_text, existing_context),
        ]

        results = await asyncio.gather(*tasks)
        embeddings = results[0]
        intent = results[1]
        surprisal_score = results[2]

        logger.info(f"🧠 Intent: {intent} | Surprisal: {surprisal_score:.2f}")

        # Early Exit for noise
        if intent == "Small Talk" and surprisal_score < 3.2:
            logger.info("⏩ Skipping Small Talk (Low Surprisal)")
            return {
                "action": "skipped",
                "reason": "small_talk",
                "score": surprisal_score,
            }

        # Step 4: Conflict Check (Only if persistent)
        similar_memories = await self.get_enhanced_search_memories(
            embeddings[0], user_id, user_handle=user_handle
        )
        conflict_info = await self.detect_conflict(memory_text, similar_memories)

        force_store = False
        if conflict_info:
            c_type = conflict_info.get(
                "type", "UPDATE"
            )  # Default to UPDATE if not specified
            logger.info(f"🚀 CONFLICT DETECTED [{c_type}]: {conflict_info['reason']}.")

            if c_type == "UPDATE":
                force_store = True  # Facts changed, force update
                try:
                    await self.qdrant_client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=models.PointIdsList(
                            points=[conflict_info["conflicting_point_id"]]
                        ),
                    )
                except Exception as e:
                    logger.error(f"Failed to delete conflict: {e}")
            elif c_type == "WARNING":
                # Do NOT delete. We want both "Arjun hates peanuts" and "Riya loves peanuts".
                # But we force store to ensure the new info is definitely kept.
                # In a future iteration, we could add a "relation" link here.
                force_store = True
                logger.info("⚠️ Safety Warning: Keeping both memories for context.")

        # Step 5: Final logic path
        score_boost = 99.9 if (force_store or intent == "Personal Fact") else 0.0

        # Parallelize Taxonomy if needed
        if not radix_path:
            radix_path = await self.get_taxonomy_path(memory_text, user_handle)

        # Step 5: Final Storage
        result = await store_neural_memory(
            user_id=user_id,
            memory_text=memory_text,
            categories=categories,
            qdrant_client=self.qdrant_client,
            surprisal_gate=self.surprisal_gate,
            taxonomy_llm=self.taxonomy_llm,
            lmcache_manager=self.lmcache_manager,
            existing_context=existing_context,
            radix_path=radix_path,
            score_boost=score_boost,
            user_handle=user_handle,
            embedding=embeddings[0],
            timestamp=timestamp,
        )

        # Step 6: Trigger Summarization Check (Radix Tree Optimization)
        if result.get("action") == "stored" and radix_path:
            # We run this in the background to not block the response
            asyncio.create_task(self.summarize_branch(radix_path, user_id, user_handle))

        return result

    async def get_taxonomy_path(self, text: str, user_handle: str = "user") -> str:
        """
        Exposes the radix path generator for taxonomy classification.
        Injects existing paths for structural grounding.
        """
        existing_paths = await self.lmcache_manager.get_all_known_paths(user_handle)
        paths = await self.taxonomy_llm.generate_radix_path(
            text, user_handle, existing_paths
        )
        return paths[
            0
        ]  # Return the first one to maintain backward compatibility for tests expecting a string

    async def get_taxonomy_paths(
        self, text: str, user_handle: str = "user", mode: str = "storage"
    ) -> List[str]:
        """
        New method to get multiple paths.
        """
        existing_paths = await self.lmcache_manager.get_all_known_paths(user_handle)
        return await self.taxonomy_llm.generate_radix_path(
            text, user_handle, existing_paths, mode=mode
        )

    async def prefetch_context(
        self, query: str, user_id: int, user_handle: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Prefetch context for inference.
        """
        return await self.inference_prefetcher.prefetch_context(
            query, user_id, user_handle
        )

    async def get_surprisal_score(self, text: str, context: str = "") -> float:
        """
        Calculates the surprisal score for the given text.
        """
        return await self.surprisal_gate.calculate_surprisal(text, context)

    async def search_recursive_context(
        self,
        radix_path: str,
        user_handle: str,
        direction: str = "up",
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Search for related contexts in the radix tree.
        """
        return await self.recursive_search.search_context(
            radix_path, user_handle, direction, max_depth
        )

    async def get_enhanced_search_memories(
        self,
        search_vector: list[float],
        user_id: int,
        categories: Optional[list[str]] = None,
        user_handle: Optional[str] = None,
        search_text: Optional[str] = None,
        **kwargs,
    ) -> List[RetrievedMemory]:
        """
        Streamlined search: High-performance retrieval with total isolation.
        Uses a single request with prefetching for speed.
        """
        # STEP A: The "Smart Searcher" - Ask the Filer where to look
        if search_text and user_handle:
            try:
                # Predict where this info would be stored - asking for multiple possibilities
                predicted_paths = await self.get_taxonomy_paths(
                    search_text, user_handle, mode="search"
                )
                logger.info(
                    f"🔮 [SMART-SEARCH] Predicted paths for '{search_text}': {predicted_paths}"
                )

                # Extract segments from all predicted paths to augment categories
                # We aggregate all valid segments from all paths to form a "Semantic Fingerprint"
                all_parts = []
                for p_path in predicted_paths:
                    parts = [
                        p
                        for p in p_path.split("/")
                        if p not in ["users", user_handle, "detail"]
                    ]

                    # FORCE ENTITY LOGIC per path
                    if "entities" in parts:
                        try:
                            idx = parts.index("entities")
                            if idx + 1 < len(parts):
                                entity_name = parts[idx + 1]
                                if entity_name not in all_parts:
                                    all_parts.append(entity_name)
                        except ValueError:
                            pass

                    for p in parts:
                        if p not in all_parts:
                            all_parts.append(p)

                if not categories:
                    categories = []

                # Merge predicted segments into categories
                for p in all_parts:
                    if p not in categories:
                        categories.append(p)

            except Exception as e:
                logger.warning(f"Smart search prediction failed: {e}")

        # LOGGING: Print what we are searching for
        search_label = f"CATEGORIES: {categories}" if categories else "PARTITION SEARCH"
        logger.info(f"🔍 [FAST-SEARCH] Query: {search_label} for {user_handle}")

        must_conditions = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        ]

        # REQUIREMENT 3: Total Isolation Fence
        if user_handle:
            must_conditions.append(
                models.FieldCondition(
                    key="user_handle", match=models.MatchValue(value=user_handle)
                )
            )

        # Build query filter
        query_filter = Filter(must=must_conditions)
        if categories:
            # The "Smart Filter" Logic: Use MatchText for partial/segment matching on paths
            # AUTO-BROADENING: We don't just match the full path, we also try to match the parent paths
            # This implements the "Top 3 most likely folders" requirement by walking up the tree

            # 1. Add the predicted (full) categories
            path_filters = [
                models.FieldCondition(
                    key="radix_paths", match=models.MatchText(text=cat)
                )
                for cat in categories
            ]

            # 2. Add 'parent' domain broadening if we have enough segments
            # e.g. if we have self/techniques/productivity/iron_vault, also check self/techniques/productivity and self/techniques
            # This helps if the specific leaf node is different (e.g. 'productivity' vs 'system')
            if len(categories) > 3:
                # Assuming standard structure: [users, handle, subject, domain, subdomain, ...]
                # We can construct broad path strings from the categories list if it was derived from a path string
                # Since 'categories' here is a flat list of path parts, let's try to reconstruct/use segments
                pass  # The MatchText on 'radix_paths' index is token-based, so searching for "techniques" ALREADY matches "self/techniques/..."
                # However, if the query predicted "gaming" but it was stored in "profile", that won't help.
                # But if the query predicted "techniques/productivity/iron_vault", it will match "techniques".

            query_filter.must.append(models.Filter(should=path_filters))

        # Execute single high-speed query
        # We use a lower threshold for category-specific search
        outs = await self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=search_vector,
            with_payload=True,
            query_filter=query_filter,
            score_threshold=0.3 if categories else 0.35,
            limit=15,
        )

        all_points = list(outs.points)

        # SPEED-PRO FALLBACK: If specific path search is sparse, broaden to user partition
        if categories and len(all_points) < 3:
            logger.info(
                "🌳 [BREADTH-SCAN] Specific path sparse. Broadening to user partition..."
            )
            # Create a broad filter that checks both 'self' and 'entities' implicitly by just filtering by user/handle
            # But we can also add a 'should' clause for 'self' and 'entities' to prioritize them if needed
            # The user requested: "check both 'self/' and 'entities/' in a single query"

            broad_outs = await self.qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=search_vector,
                with_payload=True,
                query_filter=Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=user_id)
                        ),
                        models.FieldCondition(
                            key="user_handle",
                            match=models.MatchValue(value=user_handle),
                        ),
                    ]
                    if user_handle
                    else [
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=user_id)
                        )
                    ]
                ),
                score_threshold=0.30,  # Lowered threshold slightly for broad scan to catch semantic matches
                limit=15,  # Increased limit to catch more potential matches
            )

            seen_ids = {p.id for p in all_points if p}
            for p in broad_outs.points:
                if p and p.id not in seen_ids:
                    all_points.append(p)
                    seen_ids.add(p.id)

        return [
            self._convert_enhanced_retrieved_records(point)
            for point in all_points
            if point is not None
        ]

    async def summarize_branch(
        self, radix_path: str, user_id: int, user_handle: str, threshold: int = 15
    ):
        """
        Check if a branch has too many entries and summarize them.
        Optimizes Radix Tree by collapsing noise into a single signal.
        """
        # Count points in this path
        must_conditions = [
            models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id)
            ),
            models.FieldCondition(
                key="user_handle", match=models.MatchValue(value=user_handle)
            ),
            models.FieldCondition(
                key="radix_paths", match=models.MatchText(text=radix_path)
            ),
        ]

        # We use scroll to get points
        points, next_page_offset = await self.qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=must_conditions),
            limit=threshold + 1,
            with_payload=True,
            with_vectors=False,
        )

        if len(points) <= threshold:
            return None

        logger.info(
            f"🌿 [SUMMARIZATION-TRIGGER] Branch {radix_path} has {len(points)} entries. Commencing summarization..."
        )

        # Extract text from all points
        all_texts = []
        for p in points:
            txt = p.payload.get("memory_texts")
            if txt:
                all_texts.append(txt)

        if not all_texts:
            return None

        combined_text = "\n".join(all_texts)

        # Call LLM to summarize
        prompt = f"""You are a specialized Memory Summarization Module. 
The memory branch '{radix_path}' is becoming cluttered. 
Summarize the following granular entries into a single, high-density, factual paragraph.
Focus on the most current state and persistent facts.

ENTRIES:
{combined_text}

JSON response: {{"summary": "Concise summary here"}}
"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a specialized memory summarization module for the Radix-Titan engine.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            summary_text = data.get("summary", "")

            if not summary_text:
                return None

            # 1. Delete the old granular points
            point_ids = [p.id for p in points]
            await self.qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=point_ids),
            )

            # 2. Store the summarized memory (this will create a new single point and KV cache)
            # We mark it as summarized and use store_memory recursively (without triggering summarization again immediately)
            # Actually, to avoid an infinite loop or issues, we can just call store_neural_memory directly
            await store_neural_memory(
                user_id=user_id,
                memory_text=f"[BRANCH SUMMARY]: {summary_text}",
                categories=radix_path.split("/"),
                qdrant_client=self.qdrant_client,
                surprisal_gate=self.surprisal_gate,
                taxonomy_llm=self.taxonomy_llm,
                lmcache_manager=self.lmcache_manager,
                radix_path=radix_path,
                score_boost=99.9,  # Force storage of the summary
                user_handle=user_handle,
            )

            logger.info(
                f"✅ [SUMMARIZATION-COMPLETE] {radix_path} has been collapsed into a single node."
            )
            return summary_text

        except Exception as e:
            logger.error(f"Summarization failed for branch {radix_path}: {e}")
            return None

    def _convert_enhanced_retrieved_records(self, point) -> RetrievedMemory:
        """Convert with radix_path support"""
        payload = point.payload
        radix_paths = payload.get("radix_paths")
        if not radix_paths and payload.get("radix_path"):
            radix_paths = [payload.get("radix_path")]

        return RetrievedMemory(
            point_id=point.id,
            user_id=payload["user_id"],
            memory_text=payload["memory_texts"],
            categories=payload["categories"],
            date=payload["date"],
            score=getattr(point, "score", 1.0),
            radix_path=payload.get("radix_path"),
            radix_paths=radix_paths,
            timestamp=payload.get("timestamp"),
        )

    async def get_temporal_context(
        self,
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
