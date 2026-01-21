from typing import Dict, List, Optional
import json
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv(override=True)

import asyncio
import uuid
import hashlib
from rich.console import Console
from rich.rule import Rule
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, OpenAI
import asyncio
from qdrant_client import AsyncQdrantClient, models

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from embed import generate_embeddings
from update_memory import update_memories
from vectordb import (
    get_all_categories,
    search_memories,
    stringify_retrieved_point,
    client as qdrant_client,
    COLLECTION_NAME,
    RetrievedMemory,
)
from neural_memory import NeuralMemoryEngine

console = Console(log_path=False)

# Use sync OpenAI client with asyncio.to_thread() to avoid async HTTP issues
# API key is pulled from environment variable for security
sync_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0)


async def call_openai_chat(**kwargs):
    """Wrapper to call sync OpenAI in async context"""
    return await asyncio.to_thread(sync_client.chat.completions.create, **kwargs)


async def deconstruct_query(query: str) -> List[str]:
    # SPEED optimization: Skip cloud-based deconstruction, it's too slow.
    return [query]


# Debug: Print API key status on import
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"🔑 OpenAI API Key loaded: {api_key[:10]}...")
else:
    print("❌ No OpenAI API Key found in environment!")


class ResponseGeneratorInput(BaseModel):
    """Input model for response generation"""

    transcript: list[dict] = Field(
        description="Past conversation transcript between user and AI agent"
    )
    existing_categories: list[str] = Field(
        description="List of existing categories in the memory database"
    )
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
    user_handle: str = "user",
    prefetched_memories: List[RetrievedMemory] = None,
) -> ResponseGeneratorOutput:
    """
    ReAct-like agent using OpenAI function calling
    """
    system_prompt = f"""### ROLE
You are a Context-Aware Assistant and Memory Manager for {user_handle}. You have access to their 'Neural Memory' (retrieved via Radix-Titan).

### OPERATING RULES
1. **Precision First:** When the user asks a specific question, scan the [STITCHED_MEMORIES] for the exact answer. If the answer is there, CALL finalize_response IMMEDIATELY. Do NOT call other tools unless the answer is missing.
2. **Perspective Shifting (IDENTITY PROTECTION):** Stored memories are often in the user's first person (e.g., "My manager is Rahul"). When answering, you MUST flip this to the assistant's perspective. Change "My" to "Your" and "I" to "You". Example: Respond "Your manager's name is Rahul" instead of "My manager's name is Rahul".
3. **Strict Honesty:** If the user asks for a specific fact (e.g., location, workplace, name) and it is NOT explicitly in the retrieved memories, you MUST state that you do not know. 
4. **No Topic Substitution:** If you know what the user does for a living but NOT where they live, and they ask "Where do I live?", do NOT answer with their job. Say: "I know you work as X, but I don't have a record of where you live."
5. **Memory Management:** If the user provides NEW factual information, updates to old facts, or secrets, identify this as memory-worthy. Use finalize_response with save_memory=True.
6. **No Meta-Talk:** Never say "Based on my memories" or "I found 5 results." Just answer naturally.
7. **Silence on Failure:** If the specific answer is NOT in the memories or the transcript, admit you don't know and ask for clarification.

### TOOLS
Use the fetch_similar_memories tool to search for relevant context. ALWAYS provide your final response using finalize_response."""

    # Prepare function definitions for OpenAI
    function_definitions = [
        {
            "type": "function",
            "function": {
                "name": "fetch_similar_memories",
                "description": "Essential: Search memories from vector database to find specific facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_text": {
                            "type": "string",
                            "description": "Atomic search terms extracted from user question.",
                        },
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Relevant categories to filter by.",
                        },
                    },
                    "required": ["search_text", "categories"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recursive_context_search",
                "description": "Explore related radix nodes if initial results are sparse.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "radix_path": {
                            "type": "string",
                            "description": "Path to explore.",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["up", "side"],
                            "description": "Traverse up (parent) or side (sibling).",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why more context is needed.",
                        },
                    },
                    "required": ["radix_path", "direction", "reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "forget_memories",
                "description": "Delete specific memories when user asks to forget or remove information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The specific topic to forget (e.g., 'Kerala', 'work', 'my name').",
                        },
                        "point_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: Specific point IDs to delete if already retrieved.",
                        },
                    },
                    "required": ["topic"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_response",
                "description": "Provide the final direct answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "The concise, literal answer to the user's question.",
                        },
                        "save_memory": {
                            "type": "boolean",
                            "description": "CRITICAL: Set to True if the user provided ANY new information that should be remembered permanently.",
                        },
                    },
                    "required": ["response", "save_memory"],
                },
            },
        },
    ]

    stitched_memories_text = "\n".join(
        [stringify_retrieved_point(m) for m in (prefetched_memories or [])]
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""[TRANSCRIPT]
{json.dumps(transcript, indent=2)}

[STITCHED_MEMORIES]
{stitched_memories_text if stitched_memories_text else "No initial memories found."}

Existing categories: {json.dumps(existing_categories)}

Question: {question}

Use your tools to find the answer in [STITCHED_MEMORIES] or verify it in [TRANSCRIPT].""",
        },
    ]

    for iteration in range(max_iters):
        # Check if we already have tool results - if so, skip to final response
        has_tool_results = any(
            isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages
        )

        # Determine tool choice - Force search in 1st iteration if no memories yet
        is_first_iter = iteration == 0 and not has_tool_results

        # SPEED-PRO: If we already have prefetched memories, we don't NEED to force search
        if iteration == 0 and prefetched_memories and len(prefetched_memories) > 0:
            current_tool_choice = "auto"
        else:
            current_tool_choice = "required" if is_first_iter else "auto"

        try:
            # Determine if we should stream (only on the potential final response)
            should_stream = not is_first_iter

            response = await call_openai_chat(
                model="gpt-4o-mini",
                messages=messages,
                tools=function_definitions,
                tool_choice=current_tool_choice,
                temperature=1,
                max_tokens=4000,
                stream=should_stream,
            )

            if should_stream:
                # Handle streaming response for instant feedback
                full_content = ""
                final_tool_calls = []

                # We need to collect tool calls or text
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        # For the agent, we usually expect tool calls first.
                        # If it sends content, it's a fallback.
                        full_content += delta.content
                        yield {"type": "content", "delta": delta.content}

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            # Standard tool call assembly logic (simplified for agent)
                            if len(final_tool_calls) <= tc_delta.index:
                                final_tool_calls.append(tc_delta)
                            else:
                                if tc_delta.function and tc_delta.function.arguments:
                                    final_tool_calls[
                                        tc_delta.index
                                    ].function.arguments += tc_delta.function.arguments

                # Create a mock response object that looks like the non-streamed one
                class StreamedResponse:
                    def __init__(self, content, t_calls):
                        class Message:
                            def __init__(self, c, tc):
                                self.content = c
                                self.tool_calls = tc

                        class Choice:
                            def __init__(self, c, tc):
                                self.message = Message(c, tc)

                        self.choices = [Choice(content, t_calls)]

                response = StreamedResponse(
                    full_content, final_tool_calls if final_tool_calls else None
                )

            yield {"type": "response_object", "value": response}

        except Exception as e:
            console.log(f"OpenAI API failed, using intelligent neural simulation: {e}")
            # Create intelligent mock response based on neural memory system
            response = await create_intelligent_mock_response(
                messages, function_definitions, neural_memory_engine
            )

        message = response.choices[0].message

        # Convert message to dict format for storage in messages list
        # (Real OpenAI messages are objects, MockMessages are objects too)
        if hasattr(message, "content"):
            # It's an object (OpenAI Message or MockMessage) - convert to dict
            message_dict = {
                "role": "assistant",
                "content": getattr(message, "content", None),
            }
            if hasattr(message, "tool_calls") and message.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": getattr(tc, "id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": getattr(tc.function, "name", "")
                            if hasattr(tc, "function")
                            else "",
                            "arguments": getattr(tc.function, "arguments", "{}")
                            if hasattr(tc, "function")
                            else "{}",
                        },
                    }
                    for i, tc in enumerate(message.tool_calls)
                ]
            messages.append(message_dict)
        elif isinstance(message, dict):
            messages.append(message)
        else:
            # Fallback: try to convert
            messages.append({"role": "assistant", "content": str(message)})

        # Check if the model wants to call a function
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls_processed = False
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "forget_memories":
                    topic = function_args.get("topic", "")
                    point_ids = function_args.get("point_ids", [])

                    console.log(f"🗑️ Deletion requested for topic: {topic}")

                    if not point_ids:
                        # Search for the memories to delete first
                        emb = (await generate_embeddings([topic]))[0]
                        mems = await neural_memory_engine.get_enhanced_search_memories(
                            emb, user_id, user_handle=user_handle
                        )
                        point_ids = [m.point_id for m in mems if m.score > 0.6]

                    if point_ids:
                        await qdrant_client.delete(
                            collection_name=COLLECTION_NAME,
                            points_selector=models.PointIdsList(points=point_ids),
                        )
                        tool_result = f"Successfully deleted {len(point_ids)} memories related to '{topic}'."
                    else:
                        tool_result = f"I couldn't find any specific memories to delete about '{topic}'."

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )
                    tool_calls_processed = True

                elif function_name == "fetch_similar_memories":
                    search_text = function_args.get("search_text", "")
                    if not search_text or not search_text.strip():
                        search_text = question  # Fallback to original question

                    categories = function_args.get("categories", [])

                    all_memories = []
                    seen_ids = set()

                    # EMERGENCY FLATTENED SEARCH (Sub-Second Latency Optimization)
                    # We skip multi-query deconstruction and temporal context for now to hit <2s latency
                    console.log(
                        f"⚡ [FAST-RETRIEVE] Direct vector search for: {search_text[:30]}..."
                    )

                    # Single parallel block for embedding + primary search
                    search_vector = (await generate_embeddings([search_text]))[0]
                    results = await neural_memory_engine.get_enhanced_search_memories(
                        search_vector,
                        user_id=user_id,
                        categories=None if len(categories) == 0 else categories,
                        user_handle=user_handle,
                        search_text=search_text,  # Pass text for Smart Search prediction
                    )

                    for r in results:
                        if r.point_id not in seen_ids:
                            all_memories.append(r)
                            seen_ids.add(r.point_id)

                    # Temporal search and Multi-query deconstruction are disabled for performance
                    # console.log("🕒 Skipping temporal neighbors for performance...")

                    # 4. KV CACHE STITCHING
                    extra_contexts = []
                    for m in all_memories:
                        paths = getattr(m, "radix_paths", []) or (
                            [m.radix_path] if m.radix_path else []
                        )
                        for path in paths:
                            console.log(f"🧵 Stitching KV Cache for {path}...")
                            kv_cache = await neural_memory_engine.lmcache_manager.retrieve_kv_cache(
                                path
                            )
                            if kv_cache:
                                context_header = f"### RETRIEVED NEURAL MEMORY (RADIX PATH: {path}) ###"
                                context_body = kv_cache.get("memory_text", "")
                                extra_contexts.append(
                                    f"{context_header}\n{context_body}\n"
                                )

                    # REQUIREMENT 1: Recency-Weighted Context assembly
                    # Sort memories by timestamp: Newest first
                    all_memories.sort(
                        key=lambda x: x.timestamp if x.timestamp else 0, reverse=True
                    )

                    # Explicitly find the most recent "Identity" fact
                    current_identity = next(
                        (
                            m
                            for m in all_memories
                            if any(
                                kw in m.memory_text.lower()
                                for kw in ["my name", "i am", "who am i", "call me"]
                            )
                        ),
                        None,
                    )
                    identity_context = f"CURRENT USER IDENTITY: {current_identity.memory_text if current_identity else 'Unknown'}"

                    memories_str = [
                        stringify_retrieved_point(m_) for m_ in all_memories[:10]
                    ]  # Take top 10 most recent/relevant

                    if all_memories:
                        console.log(
                            f"✅ Found {len(all_memories)} interconnected memories | Identity: {current_identity.memory_text[:20] if current_identity else 'Unknown'}"
                        )
                    else:
                        console.log("ℹ️ No matching memories found")

                    tool_result_content = {
                        "CURRENT_USER_IDENTITY": identity_context,
                        "memories": memories_str,
                    }
                    if extra_contexts:
                        # Also sort extra contexts if possible, or just append
                        tool_result_content["[STITCHED_MEMORIES]"] = extra_contexts

                    tool_result = json.dumps(tool_result_content)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )
                    tool_calls_processed = True

                elif function_name == "recursive_context_search":
                    # Handle recursive context search
                    radix_path = function_args.get("radix_path", "")
                    direction = function_args.get("direction", "up")
                    reason = function_args.get("reason", "")

                    console.log(
                        f"Recursive context search: {radix_path} ({direction}) - {reason}"
                    )

                    related_contexts = (
                        await neural_memory_engine.search_recursive_context(
                            radix_path, user_handle, direction, max_depth=3
                        )
                    )

                    # RECURSIVE UP-TREE FALLBACK (User Step 3)
                    if not related_contexts and direction == "up":
                        console.log(
                            "🌳 [ROOT-SCAN] Leaf search failed, moving up to root node for broad scan..."
                        )
                        # Search root path (simplified)
                        related_contexts = (
                            await neural_memory_engine.search_recursive_context(
                                radix_path.split("/")[0]
                                if "/" in radix_path
                                else radix_path,
                                user_handle,
                                "up",
                                max_depth=5,
                            )
                        )

                    context_summaries = []
                    for ctx in related_contexts:
                        context_summaries.append(
                            f"Related context from {ctx['radix_path']}: {ctx.get('kv_cache', {}).get('memory_text', 'N/A')}"
                        )

                    tool_result = json.dumps(
                        {
                            "related_contexts": context_summaries,
                            "direction": direction,
                            "radix_path": radix_path,
                            "was_broad_scan": not bool(
                                related_contexts
                            ),  # Indicates if it had to move up
                        }
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )
                    tool_calls_processed = True

                elif function_name == "finalize_response":
                    # Extract the final response
                    response_text = function_args.get("response", "")
                    save_memory = function_args.get("save_memory", False)
                    yield {
                        "type": "final_output",
                        "value": ResponseGeneratorOutput(
                            response=response_text, save_memory=save_memory
                        ),
                    }
                    return

            # If we processed tool calls (but didn't finalize), continue to next iteration to get final response
            if tool_calls_processed:
                continue
        else:
            # No tool calls, model provided a direct response
            # Try to extract response from message content
            if message.content:
                # If model didn't use finalize_response, we need to ask it to do so
                messages.append(
                    {
                        "role": "user",
                        "content": "Please use the finalize_response function to provide your final answer and indicate if memory should be saved.",
                    }
                )
                continue
            else:
                # Fallback: create response from last message
                yield {
                    "type": "final_output",
                    "value": ResponseGeneratorOutput(
                        response=message.content
                        or "I apologize, but I couldn't generate a proper response.",
                        save_memory=False,
                    ),
                }
                return

    # Try to get final response from OpenAI
    try:
        # Add a final instruction to be direct
        final_instruction = "Verify the facts in [STITCHED_MEMORIES] and provide a direct, concise answer to the user's question. Use finalize_response."

        final_response = await call_openai_chat(
            model="gpt-4o-mini",
            messages=messages + [{"role": "user", "content": final_instruction}],
            tools=function_definitions,
            tool_choice={"type": "function", "function": {"name": "finalize_response"}},
            temperature=1,
            max_tokens=4000,
        )
    except Exception as e:
        console.log(f"OpenAI API failed for final response, using neural fallback: {e}")
        final_response = await create_final_mock_response(
            messages, neural_memory_engine, user_id=user_id, user_handle=user_handle
        )

    final_message = final_response.choices[0].message
    if final_message.tool_calls:
        for tool_call in final_message.tool_calls:
            if tool_call.function.name == "finalize_response":
                function_args = json.loads(tool_call.function.arguments)
                yield {
                    "type": "final_output",
                    "value": ResponseGeneratorOutput(
                        response=function_args.get("response", ""),
                        save_memory=function_args.get("save_memory", False),
                    ),
                }
                return

    # Ultimate fallback
    yield {
        "type": "final_output",
        "value": ResponseGeneratorOutput(
            response="I apologize, but I encountered an issue generating a response.",
            save_memory=False,
        ),
    }
    return


async def create_intelligent_mock_response(
    messages, function_definitions, neural_memory_engine
):
    """
    Create an intelligent mock response that simulates GPT-3.5-turbo behavior
    using the neural memory system for context awareness.
    """
    # Extract user message and conversation context
    # Filter to only dict messages (skip MockMessage objects)
    user_message = ""
    conversation_history = []

    for msg in messages:
        # Only process dict messages, skip MockMessage objects
        if not isinstance(msg, dict):
            continue

        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            conversation_history.append(f"User: {msg.get('content', '')}")
        elif msg.get("role") == "assistant":
            conversation_history.append(f"Assistant: {msg.get('content', '')}")

    # ALWAYS search memory for EVERY query (user requirement)
    tool_calls = []
    if neural_memory_engine and user_message:
        try:
            # Always trigger memory search tool for every query
            class MockToolCall:
                def __init__(self):
                    self.id = "memory_search_001"
                    self.function = MockFunction()

            class MockFunction:
                def __init__(self):
                    # Clean message for JSON
                    clean_msg = user_message.replace('"', '\\"').replace("\n", " ")[
                        :100
                    ]
                    self.name = "fetch_similar_memories"
                    self.arguments = (
                        f'{{"search_text": "{clean_msg}", "categories": []}}'
                    )

            tool_calls.append(MockToolCall())
        except:
            pass

    # Generate simple response that indicates we're searching memories
    # The actual response will come from create_final_mock_response after tool results
    response_content = f"Searching my memory for information about: '{user_message[:80]}{'...' if len(user_message) > 80 else ''}'..."

    # Create response object matching OpenAI format
    class MockChoice:
        def __init__(self, content, tool_calls):
            self.message = MockMessage(content, tool_calls)

    class MockMessage:
        def __init__(self, content, tool_calls):
            self.content = content
            self.tool_calls = tool_calls

    class MockResponse:
        def __init__(self, content, tool_calls):
            self.choices = [MockChoice(content, tool_calls)]

    return MockResponse(response_content, tool_calls if tool_calls else None)


async def create_final_mock_response(
    messages, neural_memory_engine, user_id=1, user_handle="user"
):
    """
    Create intelligent final response that ALWAYS searches memories and responds accordingly.
    """
    # Filter out mock message objects and get real messages
    real_messages = []
    for msg in messages:
        if hasattr(msg, "role") and hasattr(msg, "content"):  # Real message
            real_messages.append(msg)
        elif isinstance(msg, dict):  # Dict message
            real_messages.append(msg)

    # Extract latest user message
    user_messages = [
        msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        for msg in real_messages
        if (isinstance(msg, dict) and msg.get("role") == "user")
        or (hasattr(msg, "role") and getattr(msg, "role") == "user")
    ]

    latest_user_message = user_messages[-1] if user_messages else "General conversation"

    # ALWAYS search memories for EVERY query (both questions and statements)
    retrieved_memories = []

    # First check tool results
    for msg in real_messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            try:
                tool_content = msg.get("content", "")
                if isinstance(tool_content, str):
                    tool_data = json.loads(tool_content)
                else:
                    tool_data = tool_content

                if "memories" in tool_data and isinstance(tool_data["memories"], list):
                    retrieved_memories.extend(tool_data["memories"])
            except:
                pass

    # If no memories from tools, search directly in neural memory
    if not retrieved_memories and neural_memory_engine:
        try:
            from embed import generate_embeddings

            search_vector = (await generate_embeddings([latest_user_message]))[0]
            memories = await neural_memory_engine.get_enhanced_search_memories(
                search_vector,
                user_id=user_id,
                user_handle=user_handle,  # REQUIREMENT 3: Hard Metadata Filter
            )

            if memories:
                # REQUIREMENT 1: Sort by timestamp if available
                memories.sort(
                    key=lambda x: x.timestamp if x.timestamp else 0, reverse=True
                )

                # Convert to string format
                from vectordb import stringify_retrieved_point

                for mem in memories[:5]:  # Top 5 most recent
                    retrieved_memories.append(stringify_retrieved_point(mem))
        except Exception as e:
            console.log(f"Memory search error: {e}")

    # Determine if it's a question
    is_question = latest_user_message.strip().endswith("?") or any(
        latest_user_message.lower().strip().startswith(qword)
        for qword in ["what", "when", "where", "who", "how", "why", "which"]
    )

    # Process retrieved memories - extract pure memory text from stored memories
    relevant_facts = []
    if retrieved_memories:
        for mem_str in retrieved_memories:
            # Extract memory text (before " (Categories:")
            memory_text = (
                mem_str.split(" (Categories:")[0]
                if " (Categories:" in mem_str
                else mem_str
            )

            # Check relevance based on user query keywords only (no hardcoded keywords)
            query_keywords = set(latest_user_message.lower().split())
            memory_keywords = set(memory_text.lower().split())

            # If significant word overlap, it's relevant
            overlap = len(query_keywords & memory_keywords)
            if overlap >= 2:  # At least 2 words in common
                relevant_facts.append(memory_text)

    # Generate response based on whether we found relevant memories
    if relevant_facts:
        # Found memories - try to be direct even in fallback
        final_response_text = "\n".join(relevant_facts[:2])
        if len(relevant_facts) > 2:
            final_response_text += (
                "\n(Note: Multiple related records found in neural memory.)"
            )
        save_memory = False
    else:
        # No memories found - say we don't have info, but still store the question/info
        if is_question:
            final_response_text = (
                f"I don't have any information related to that topic in my memory."
            )
        else:
            final_response_text = f"I understand: '{latest_user_message[:100]}{'...' if len(latest_user_message) > 100 else ''}'"

        save_memory = True  # Store the question/info even if no matching memories found

    # Create proper finalize_response tool call
    class MockToolCall:
        def __init__(self, response_text, save_mem):
            self.id = "final_response_call"
            self.function = MockFunction(response_text, save_mem)

    class MockFunction:
        def __init__(self, response_text, save_mem):
            # Escape quotes properly for JSON
            safe_text = (
                response_text.replace('"', '\\"').replace("\n", "\\n").replace("\r", "")
            )
            self.name = "finalize_response"
            self.arguments = (
                f'{{"response": "{safe_text}", "save_memory": {str(save_mem).lower()}}}'
            )

    class MockChoice:
        def __init__(self, response_text, save_mem):
            self.message = MockMessage(response_text, save_mem)

    class MockMessage:
        def __init__(self, response_text, save_mem):
            self.tool_calls = [MockToolCall(response_text, save_mem)]

    class MockResponse:
        def __init__(self, response_text, save_mem):
            self.choices = [MockChoice(response_text, save_mem)]

    return MockResponse(final_response_text, save_memory)


async def run_chat(user_id, user_handle=None):
    if user_handle is None:
        # Create a stable, unique ID for the user to ensure persistence across sessions
        # while maintaining a non-predictable "unique id" format.
        user_handle = hashlib.sha256(f"user_salt_{user_id}".encode()).hexdigest()[:16]

    # Cache categories to avoid fetching them from DB on every prompt
    category_cache = await get_all_categories(user_id=user_id, user_handle=user_handle)

    past_messages = []

    # Initialize Neural Memory Engine
    neural_memory_engine = NeuralMemoryEngine()
    # Pass OpenAI client for conflict detection and intent classification
    await neural_memory_engine.initialize(qdrant_client, sync_client)

    console.print("Initializing Radix-Titan Neural Memory Engine...", style="bold blue")
    console.print("Let's begin to chat!", style="bold green")

    while True:
        question = console.input("[bold cyan]> [/bold cyan]")
        console.print(Rule(style="grey50"))

        with console.status("[bold green] Working..."):
            # Update transcript BEFORE calling agent
            current_user_message = {"role": "user", "content": question}
            past_messages.append(current_user_message)
            transcript_window = past_messages[-10:]

            async def get_multi_prefetch(q, uid, uhandle):
                # 1. Topic Splitting (Atomic Separation)
                # Split by 'and', 'also', 'furthermore' for parallel searching
                sub_queries = [
                    s.strip()
                    for s in q.replace(" and ", "|")
                    .replace(" also ", "|")
                    .replace("?", "")
                    .split("|")
                ]
                if len(sub_queries) == 1:
                    # Plus a broad version
                    sub_queries.append(f"identity and personal background of {uhandle}")

                # 2. Parallel Embedding Cluster
                emb_res = await generate_embeddings(sub_queries)

                # 3. Parallel Vector Burst
                search_tasks = [
                    neural_memory_engine.get_enhanced_search_memories(
                        emb, uid, user_handle=uhandle
                    )
                    for emb in emb_res
                ]
                results = await asyncio.gather(*search_tasks)

                # Flatten and deduplicate by point ID
                all_m = []
                seen_ids = set()
                for cluster in results:
                    for m in cluster:
                        if m.point_id not in seen_ids:
                            all_m.append(m)
                            seen_ids.add(m.point_id)
                return all_m

            # Kick off the retrieval burst
            memories_res = await get_multi_prefetch(question, user_id, user_handle)

        # Exit status and start streaming
        final_response_obj = None
        any_content_streamed = False

        async for chunk in react_agent(
            user_id=user_id,
            transcript=transcript_window,
            question=question,
            existing_categories=category_cache,
            tools={},
            neural_memory_engine=neural_memory_engine,
            max_iters=2,
            user_handle=user_handle,
            prefetched_memories=memories_res,
        ):
            if chunk["type"] == "content":
                if not any_content_streamed:
                    console.print("[bold green]AI:[/bold green] ", end="")
                    any_content_streamed = True
                console.print(chunk["delta"], end="")
            elif chunk["type"] == "final_output":
                final_response_obj = chunk["value"]

        # If we got a final response but nothing was streamed (1-shot response)
        if final_response_obj and not any_content_streamed:
            console.print("[bold green]AI:[/bold green] ", end="")
            console.print(final_response_obj.response)
        elif not any_content_streamed:
            console.print(
                "[bold red]AI: I encountered an unexpected error processing your request.[/bold red]"
            )
            final_response_obj = ResponseGeneratorOutput(
                response="Error", save_memory=False
            )

        console.print()

        if not final_response_obj:
            # Fallback if AI didn't call finalize_response
            final_response_obj = ResponseGeneratorOutput(
                response="I'm sorry, I couldn't process that.", save_memory=False
            )

        response = final_response_obj.response
        out = final_response_obj

        # Complete the transcript with assistant response
        past_messages.append({"role": "assistant", "content": response})

        if out.save_memory:
            # Extract user message for memory storage
            user_message = ""
            for msg in past_messages[-2:]:  # Last 2 messages
                if msg["role"] == "user":
                    user_message = msg["content"]

            # Use neural memory engine for storage (IT RUNS AS A BACKGROUND TASK)
            console.log(f"🧠 Persisting memory in background...")

            # Internal helper to handle the task safely
            async def store_task_wrapper(msg, uid, uhandle):
                try:
                    await neural_memory_engine.store_memory(
                        user_id=uid,
                        memory_text=msg,
                        categories=[],
                        existing_context="\n".join(
                            [msg["content"] for msg in past_messages[:-2]]
                        ),
                        user_handle=uhandle,
                    )
                except Exception as e:
                    logger.error(f"Background Storage Error: {e}")

            # Fire and forget
            asyncio.create_task(store_task_wrapper(user_message, user_id, user_handle))

            # Update categories in background too
            async def update_cats():
                nonlocal category_cache
                new_cats = await get_all_categories(
                    user_id=user_id, user_handle=user_handle
                )
                category_cache = new_cats

            asyncio.create_task(update_cats())

        # Final spacing
        console.print()
        console.print(Rule(style="grey50"))


if __name__ == "__main__":
    asyncio.run(run_chat(1))
