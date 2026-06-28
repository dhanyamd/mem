from typing import Dict, List, Optional, Any
import json
import os
import asyncio
import random
import re
from datetime import datetime
from rich.console import Console
from pydantic import BaseModel
from openai import AsyncOpenAI
from embed import generate_embeddings
from neural_memory import NeuralMemoryEngine

console = Console(log_path=False)
client_openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=2)

class ResponseGeneratorOutput(BaseModel):
    response: str
    save_memory: bool

class ResponseGenerator:
    def __init__(self, engine: NeuralMemoryEngine):
        self.engine = engine

    async def call_openai_chat(self, messages: List[Dict[str, str]]):
        """Routing logic: Pure OpenAI with clean client-side retries."""
        try:
            resp = await client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                timeout=15.0
            )
            return resp
        except Exception:
            # Fallback to 4o only on total mini failure
            try:
                resp = await client_openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.1,
                    timeout=25.0
                )
                return resp
            except Exception as e:
                print(f"❌ OpenAI Total Failure: {e}")
                raise e

    async def generate_response(self, query: str, user_id: int, user_handle: str = None, **kwargs) -> str:
        if not user_handle:
            user_handle = str(user_id)
            
        memory_context = ""
        try:
            vectors = await generate_embeddings([query])
            if vectors is not None and len(vectors) > 0:
                search_vector = vectors[0]
                
                memories = await self.engine.get_enhanced_search_memories(
                    search_vector=search_vector,
                    user_id=int(user_id),
                    conversation_id=kwargs.get("conversation_id"),
                    user_handle=user_handle,
                    search_text=query
                )
                
                # 3. 🕰️ CHRONO-RE-RANKING (April Bug Neutralization)
                def extract_date(text):
                    # Match dates like [10:43 am on 4 February, 2023]
                    match = re.search(r'on (\d{1,2} [A-Z][a-z]+, \d{4})', text)
                    if match:
                        try:
                            # Handle both '4' and '04' for day
                            return datetime.strptime(match.group(1), "%d %B, %Y")
                        except: 
                            try:
                                # Fallback for different formats
                                return datetime.strptime(match.group(1).strip(), "%d %B %Y")
                            except: return datetime(1970, 1, 1)
                    return datetime(1970, 1, 1)

                sorted_mems = sorted(memories[:151], key=lambda x: extract_date(x.memory_text))
                memory_context = "\n".join([f"- {m.memory_text}" for m in sorted_mems])
        except Exception as e:
            print(f"⚠️ Search failed: {e}")
            memory_context = "No memories retrieved."

        system_prompt = """
        You are the Radical-Titan Neural Memory Engine. Your goal is to answer questions about the user's history with 100% chronological accuracy.

        CORE REASONING RULES:
        1. Every memory is prefixed with its absolute session date in brackets [D Month, YYYY].
        2. For "When did X first happen?" (Historical Onset), search all sessions and prioritize the EARLIEST date mentioned.
        3. For current status/properties, prioritize the LATEST date.
        4. If there is a contradiction (e.g., "Jon lost his job" in Jan vs "Jon is working" in Aug), use the date branding to resolve it.

        OUTPUT GUIDELINES:
        - Answer in a concise, natural conversational tone.
        - If the user asks for a date, use the format "D Month, YYYY" (e.g., "7 May, 2023").
        - If the user asks "what" or "why", provide a full natural language answer based on the context.
        - If the information is not present, say "I don't have information about that."
        """
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"RETRIEVED MEMORIES:\n{memory_context}\n\nQ: {query}"}
        ]

        # 📸 TRUTH CAPTURE: Log the prompt for offline debugging
        try:
            with open("/tmp/titan_debug_prompt.txt", "a") as f:
                f.write(f"\n\n=== QUESTION: {query} ===\n")
                f.write(f"PROMPT:\n{messages[1]['content']}\n")
                f.write("====================================\n")
        except: pass

        try:
            response = await self.call_openai_chat(messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error Generating Response: {e}"

async def react_agent(user_id, transcript, question, *args, **kwargs):
    engine = kwargs.get("engine")
    if not engine:
        from qdrant_client import AsyncQdrantClient
        engine = NeuralMemoryEngine()
        q_client = AsyncQdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        await engine.initialize(q_client, AsyncOpenAI())
    
    gen = ResponseGenerator(engine)
    response_text = await gen.generate_response(
        query=question, 
        user_id=user_id, 
        user_handle=kwargs.get("user_handle"),
        conversation_id=kwargs.get("conversation_id")
    )
    
    yield {
        "type": "final_output",
        "value": ResponseGeneratorOutput(response=response_text, save_memory=False)
    }

# 🛠️ BACKWARD COMPATIBILITY: Module-level exports
async def call_openai_chat(messages: List[Dict[str, str]] = None, prompt: str = None, **kwargs):
    """Bridge to the Class-based routing logic"""
    if not messages and prompt:
        messages = [{"role": "user", "content": prompt}]
    
    # Create an ephemeral generator for the call if engine not needed
    gen = ResponseGenerator(None)
    return await gen.call_openai_chat(messages)

def call_openai_chat_sync(prompt: str = None, **kwargs):
    """Synchronous bridge"""
    messages = kwargs.get("messages", [])
    if not messages and prompt:
        messages = [{"role": "user", "content": prompt}]
    return asyncio.run(call_openai_chat(messages))
