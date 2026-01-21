from datetime import datetime
from pydoc import cli
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.grpc import Points, ScoredPoint
from qdrant_client.models import Distance, Filter, VectorParams, models
import asyncio
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
COLLECTION_NAME = "cognitive_memory"

class EmbeddedMemory(BaseModel):
    user_id: int
    memory_text : str
    categories: list[str]
    date: str
    embedding: list[float] 

class RetrievedMemory(BaseModel):
    point_id: str
    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    score: float
    radix_path: Optional[str] = None
    radix_paths: Optional[list[str]] = None
    timestamp: Optional[float] = None

async def create_memory_collection(): 
    if not (await client.collection_exists(COLLECTION_NAME)): 
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=64, distance=Distance.COSINE)
        )
        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.INTEGER
        )
        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_handle",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="categories",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="timestamp",
            field_schema=models.PayloadSchemaType.FLOAT,
        )
        print("Collection created with user_handle and timestamp indices")
    else:
        print("Collection exists")

async def insert_memories(memories: list[EmbeddedMemory]):
    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=uuid4().hex,
                payload={
                    "user_id": memory.user_id,
                    "categories": memory.categories,
                    "memory_texts": memory.memory_text,
                    "date": memory.date,
                },
                vector=memory.embedding,
            )
            for memory in memories
        ],
    )


async def search_memories(
    search_vector: list[float],
    user_id: int,
    categories: Optional[list[str]] = None,
    user_handle: Optional[str] = None,
):
    must_conditions: list[models.Condition] = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
    ]
    if user_handle:
        must_conditions.append(
            models.FieldCondition(key="user_handle", match=models.MatchValue(value=user_handle))
        )
    if categories is not None:
        if len(categories) > 0:
            must_conditions.append(
                models.FieldCondition(
                    key="categories", match=models.MatchAny(any=categories)
                )
            )
    outs = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=search_vector,
        with_payload=True,
        query_filter=Filter(must=must_conditions),
        score_threshold=0.1,
        limit=2,
    )

    return [
        convert_retrieved_records(point) for point in outs.points if point is not None
    ]


async def delete_user_records(user_id):
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    )
                ]
            )
        ),
    )


async def delete_records(point_ids):
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=point_ids),
    )


async def fetch_all_user_records(user_id, user_handle=None):
    must_conditions = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
    ]
    if user_handle:
        must_conditions.append(
            models.FieldCondition(key="user_handle", match=models.MatchValue(value=user_handle))
        )

    out = await client.query_points(
        collection_name=COLLECTION_NAME,
        query_filter=models.Filter(must=must_conditions),
    )
    return [convert_retrieved_records(point) for point in out.points]


def convert_retrieved_records(point) -> RetrievedMemory:
    payload = point.payload
    # Handle single vs multiple paths
    radix_paths = payload.get("radix_paths")
    if not radix_paths and payload.get("radix_path"):
        radix_paths = [payload.get("radix_path")]
        
    return RetrievedMemory(
        point_id=point.id,
        user_id=payload["user_id"],
        memory_text=payload["memory_texts"],
        categories=payload["categories"],
        date=payload["date"],
        score=point.score,
        radix_path=payload.get("radix_path"),
        radix_paths=radix_paths,
        timestamp=payload.get("timestamp")
    )


async def get_all_categories(user_id, user_handle=None):
    """
    Uses Qdrant's facet feature to efficiently get all unique categories
    from the indexed 'categories' field.
    """
    must_conditions = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
    ]
    if user_handle:
        must_conditions.append(
            models.FieldCondition(key="user_handle", match=models.MatchValue(value=user_handle))
        )

    facet_filter = Filter(must=must_conditions)

    # Use the facet method to get unique values from the indexed field
    facet_result = await client.facet(
        collection_name=COLLECTION_NAME,
        key="categories",
        facet_filter=facet_filter,
        limit=1000,  # Maximum number of unique categories to return
    )

    # Use the facet method to get unique values from the indexed field
    facet_result = await client.facet(
        collection_name=COLLECTION_NAME,
        key="categories",
        facet_filter=facet_filter,
        limit=1000,  # Maximum number of unique categories to return
    )

    unique_categories = [hit.value for hit in facet_result.hits]

    return unique_categories


def stringify_retrieved_point(retrieved_memory: RetrievedMemory):
    path_info = f" [Path: {retrieved_memory.radix_path}]" if retrieved_memory.radix_path else "" 
    return f"""{retrieved_memory.memory_text} (Categories: {retrieved_memory.categories}){path_info} Relevance: {retrieved_memory.score:.2f}"""


if __name__ == "__main__":
    asyncio.run(create_memory_collection())