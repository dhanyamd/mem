from typing import Dict, List, Optional
import json
import os
import asyncio
from rich.console import Console
from rich.rule import Rule
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

import warnings

warnings.filterwarnings("ignore")

from embed import generate_embeddings
from update_memory import update_memories
from vectordb import get_all_categories, search_memories, stringify_retrieved_point, client as qdrant_client
from neural_memory import NeuralMemoryEngine

console = Console(log_path=False)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ResponseGeneratorInput(BaseModel):
    """Input model for response generation"""
    transcript: list[dict] = Field(description="Past conversation transcript between user and AI agent")
    existing_categories: list[str] = Field(description="List of existing categories in the memory database")
    question: str = Field(description="Latest question by the user")


class ResponseGeneratorOutput(BaseModel):
    """Output model for response generation"""
    response: str = Field(description="The final response to the user")
    save_memory: bool = Field(
        description="True if a new memory record needs to be created for the latest interaction"
    )


async def react_agent(
    user_id: int,
    transcript: list[dict],
    question: str,
    existing_categories: list[str],
    tools: dict,
    neural_memory_engine: NeuralMemoryEngine,
    max_iters: int = 2,
) -> ResponseGeneratorOutput:
    """
    ReAct-like agent using OpenAI function calling
    """
    system_prompt = """You will be given a past conversation transcript between user and an AI agent. Also the latest question by the user.

You have access to a neural memory system with advanced context retrieval. Use the fetch_similar_memories tool to search for relevant context from the vector database. If the retrieved information doesn't fully address the user's question, you can use the recursive_context_search tool to explore related contexts in the knowledge tree.

Available tools:
- fetch_similar_memories: Search for memories by vector similarity and categories
- recursive_context_search: When you need broader or related context, traverse the knowledge tree upwards (parent contexts) or sideways (sibling contexts) to find additional relevant information

The neural memory system automatically manages context through a Radix-Titan architecture, providing instant access to related information through KV cache stitching.

You must output the final response, and also decide if the latest interaction needs to be recorded into the memory database. The system uses surprisal-based gating to automatically determine if new information should be stored.

New memories should be made when the USER provides new info. It is not to save information about the AI or the assistant.

When you have the final answer, use the finalize_response function to provide your response and indicate if memory should be saved."""

    # Prepare function definitions for OpenAI
    function_definitions = [
        {
            "type": "function",
            "function": {
                "name": "fetch_similar_memories",
                "description": "Search memories from vector database if conversation requires additional context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_text": {
                            "type": "string",
                            "description": "The string to embed and do vector similarity search"
                        },
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of strings taken from existing_categories. Use an empty list ([]) if you want to search across all categories."
                        }
                    },
                    "required": ["search_text", "categories"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "recursive_context_search",
                "description": "Explore related contexts in the knowledge tree when you need broader or deeper information. Use this when initial search results are insufficient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "radix_path": {
                            "type": "string",
                            "description": "The radix path from a previous memory search to explore related contexts"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["up", "side"],
                            "description": "Direction to traverse: 'up' for parent contexts (broader), 'side' for sibling contexts (related)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief explanation of why you need additional context"
                        }
                    },
                    "required": ["radix_path", "direction", "reason"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_response",
                "description": "Call this when you have the final answer to provide your response and indicate if memory should be saved.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "The final response to the user"
                        },
                        "save_memory": {
                            "type": "boolean",
                            "description": "True if a new memory record needs to be created for the latest interaction"
                        }
                    },
                    "required": ["response", "save_memory"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Transcript: {json.dumps(transcript, indent=2)}
Existing categories: {json.dumps(existing_categories)}
Question: {question}

Use the tools available to search for relevant memories if needed, then provide your final response using finalize_response."""
        }
    ]

    for iteration in range(max_iters):
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=function_definitions,
            tool_choice="auto",
            temperature=1,
            max_tokens=16000,
        )

        message = response.choices[0].message
        messages.append(message)

        # Check if the model wants to call a function
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "fetch_similar_memories":
                    # Call the actual tool function
                    search_text = function_args.get("search_text", "")
                    categories = function_args.get("categories", [])
                    
                    console.log("Search text: ", search_text)
                    console.log("Categories: ", categories)

                    search_vector = (await generate_embeddings([search_text]))[0]
                    memories = await search_memories(
                        search_vector,
                        user_id=user_id,
                        categories=None if len(categories) == 0 else categories,
                    )
                    memories_str = [stringify_retrieved_point(m_) for m_ in memories]
                    console.log(f"Retrieved memories: \n", "\n- ".join(memories_str))
                    
                    tool_result = json.dumps({"memories": memories_str})
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                elif function_name == "recursive_context_search":
                    # Handle recursive context search
                    radix_path = function_args.get("radix_path", "")
                    direction = function_args.get("direction", "up")
                    reason = function_args.get("reason", "")

                    console.log(f"Recursive context search: {radix_path} ({direction}) - {reason}")

                    related_contexts = await neural_memory_engine.search_recursive_context(
                        radix_path, direction, max_depth=3
                    )

                    context_summaries = []
                    for ctx in related_contexts:
                        context_summaries.append(f"Related context from {ctx['radix_path']}: {ctx.get('kv_cache', {}).get('memory_text', 'N/A')}")

                    tool_result = json.dumps({
                        "related_contexts": context_summaries,
                        "direction": direction,
                        "radix_path": radix_path
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                elif function_name == "finalize_response":
                    # Extract the final response
                    response_text = function_args.get("response", "")
                    save_memory = function_args.get("save_memory", False)
                    return ResponseGeneratorOutput(
                        response=response_text,
                        save_memory=save_memory
                    )
        else:
            # No tool calls, model provided a direct response
            # Try to extract response from message content
            if message.content:
                # If model didn't use finalize_response, we need to ask it to do so
                messages.append({
                    "role": "user",
                    "content": "Please use the finalize_response function to provide your final answer and indicate if memory should be saved."
                })
                continue
            else:
                # Fallback: create response from last message
                return ResponseGeneratorOutput(
                    response=message.content or "I apologize, but I couldn't generate a proper response.",
                    save_memory=False
                )

    # If we've exhausted iterations, try to get a final response
    final_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages + [{"role": "user", "content": "Please provide your final response using the finalize_response function."}],
        tools=function_definitions,
        tool_choice={"type": "function", "function": {"name": "finalize_response"}},
        temperature=1,
        max_tokens=16000,
    )

    final_message = final_response.choices[0].message
    if final_message.tool_calls:
        for tool_call in final_message.tool_calls:
            if tool_call.function.name == "finalize_response":
                function_args = json.loads(tool_call.function.arguments)
                return ResponseGeneratorOutput(
                    response=function_args.get("response", ""),
                    save_memory=function_args.get("save_memory", False)
                )

    # Ultimate fallback
    return ResponseGeneratorOutput(
        response="I apologize, but I encountered an issue generating a response.",
        save_memory=False
    )


async def run_chat(user_id):
    past_messages = []
    existing_categories = await get_all_categories(user_id=user_id)

    # Initialize Neural Memory Engine
    neural_memory_engine = NeuralMemoryEngine()
    await neural_memory_engine.initialize(qdrant_client, client)

    console.print("Initializing Radix-Titan Neural Memory Engine...", style="bold blue")
    console.print("Let's begin to chat!", style="bold green")

    while True:
        question = console.input("[bold cyan]> [/bold cyan]")
        console.print(Rule(style="grey50"))

        with console.status("[bold green] Working..."):
            out = await react_agent(
                user_id=user_id,
                transcript=past_messages,
                question=question,
                existing_categories=existing_categories,
                tools={},
                neural_memory_engine=neural_memory_engine,
                max_iters=2,
            )

            response = out.response

            past_messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response},
                ]
            )

            if out.save_memory:
                # Use neural memory engine for storage
                console.log("Storing neural memory...")

                # Extract user message for memory storage
                user_message = ""
                ai_response = ""
                for msg in past_messages[-2:]:  # Last 2 messages
                    if msg["role"] == "user":
                        user_message = msg["content"]
                    elif msg["role"] == "assistant":
                        ai_response = msg["content"]

                # Combine for context
                memory_text = f"User: {user_message}\nAssistant: {ai_response}"

                # Extract existing context for surprisal check
                existing_context = "\n".join([msg["content"] for msg in past_messages[:-2]])

                try:
                    memory_result = await neural_memory_engine.store_memory(
                        user_id=user_id,
                        memory_text=memory_text,
                        categories=[],  # Could be enhanced to extract categories
                        existing_context=existing_context
                    )
                    console.log(f"Neural memory result: {memory_result}", style="italic")
                except Exception as e:
                    console.log(f"Neural memory storage failed: {e}", style="red")

                # Refresh the existing categories that's searchable
                existing_categories = await get_all_categories(user_id=user_id)

        console.print(f"\nAI: {response}\n\n", style="bold green")