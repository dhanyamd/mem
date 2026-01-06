import asyncio
from email import message
import json
import os
from typing import List, Dict
from embed import generate_embeddings
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

class MemoryExtractRequest(BaseModel):
    """Request model for memory extraction"""
    transcript: str = Field(description="The conversation transcript to extract memories from")

class MemoryExtractResponse(BaseModel):
    """Response model containing extracted memory information"""
    information: str = Field(description="Extracted relevant information from the conversation that should be remembered")
    predicted_categories=Field(description="Extracted categories will be categorized into sad, happy or angry according to the information extracted from the conversation")

async def extract_memories_from_messages(messages: List[Dict[str, str]]) -> MemoryExtractResponse:
    """Extract relevant information from the conversation. Create memory entries that you should remember when 
    speaking to the user later.
    """
    transcript = json.dumps(messages)
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create the prompt
    prompt = f"""Extract relevant information from the conversation. Create memory entries that you should remember when 
speaking to the user later.

Conversation transcript:
{transcript}

Extract the key information that should be remembered about the user. Return a JSON object with an "information" field containing the extracted information."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[
            {"role": "system", "content": "You are a memory extraction system. Extract key information from conversations that should be remembered. Always return valid JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}  # Request JSON response
    )
    
    # Parse the response
    content = response.choices[0].message.content
    try:
        result = json.loads(content)
        information = result.get("information", content)
    except json.JSONDecodeError:
        information = content
    
    # Create and return pydantic model
    memory_response = MemoryExtractResponse(information=information)
    return memory_response 

async def embed_memories(memories: list[MemoryExtractResponse]): 
    memory_texts = [
        m.information for m in memories
    ]
    embeddings = await generate_embeddings(memory_texts)
    return embeddings

async def extract_and_embed_memories(messages, existing_categories):
    memories = await extract_memories_from_messages(messages, existing_categories) 
    embeddings = generate_embeddings(memories) 
    return memories, embeddings


if __name__ == "__main__": 
    messages = [
        {
            "role": "user",
            "content": "i like coffee"
        },
        {
            "role": "assistant",
            "content": "Got it"
        }, {
            "role": "user",
            "content": "actually, no i like tea more"
        }
    ]
    result = asyncio.run(extract_memories_from_messages(messages))
    print(result)