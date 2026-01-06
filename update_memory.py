import json
import os
from pydantic import BaseModel, Field
from datetime import datetime
from openai import AsyncOpenAI
from embed import generate_embeddings
from vectordb import (
    EmbeddedMemory,
    RetrievedMemory,
    delete_records,
    fetch_all_user_records,
    insert_memories,
    search_memories,
)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class MemoryWithIds(BaseModel):
    memory_id: int
    memory_text: str
    memory_categories: list[str]


class UpdateMemoryInput(BaseModel):
    """Input model for memory update"""
    messages: list[dict] = Field(description="Conversation between user and assistant")
    existing_memories: list[MemoryWithIds] = Field(description="Similar memories from the database")


class UpdateMemoryOutput(BaseModel):
    """Output model for memory update"""
    summary: str = Field(description="Summarize what you did. Very short (less than 10 words)")


async def update_memories_agent(
    user_id: int, messages: list[dict], existing_memories: list[RetrievedMemory]
):

    def get_point_id_from_memory_id(memory_id):
        return existing_memories[memory_id].point_id

    memory_ids = [
        MemoryWithIds(
            memory_id=idx, memory_text=m.memory_text, memory_categories=m.categories
        )
        for idx, m in enumerate(existing_memories)
    ]

    system_prompt = """You will be given the conversation between user and assistant and some similar memories from the database. Your goal is to decide how to combine the new memories into the database with the existing memories.

Actions meaning:
- ADD: add new memories into the database as a new memory
- UPDATE: update an existing memory with richer information.
- DELETE: remove memory items from the database that aren't required anymore due to new information
- NOOP: No need to take any action

If no action is required you can finish.

Think less and do actions."""

    # Prepare function definitions for OpenAI
    function_definitions = [
        {
            "type": "function",
            "function": {
                "name": "add_memory",
                "description": "Add the new_memory into the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_text": {
                            "type": "string",
                            "description": "The memory text to add"
                        },
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Categories for the memory"
                        }
                    },
                    "required": ["memory_text", "categories"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update",
                "description": "Updating memory_id to use updated_memory_text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "integer",
                            "description": "Integer index of the memory to replace"
                        },
                        "updated_memory_text": {
                            "type": "string",
                            "description": "Simple atomic factoid to replace the old memory with the new memory"
                        },
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Use existing categories or create new ones if required"
                        }
                    },
                    "required": ["memory_id", "updated_memory_text", "categories"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delete",
                "description": "Remove these memory_ids from the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of memory IDs to delete"
                        }
                    },
                    "required": ["memory_ids"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "noop",
                "description": "Call this if no action is required",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finalize",
                "description": "Call this when you're done with all actions to provide a summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Summarize what you did. Very short (less than 10 words)"
                        }
                    },
                    "required": ["summary"]
                }
            }
        }
    ]

    # Serialize memory_ids to dict format
    memory_dicts = []
    for m in memory_ids:
        if hasattr(m, 'model_dump'):
            memory_dicts.append(m.model_dump())
        else:
            memory_dicts.append(m.dict())
    
    user_prompt = f"""Messages: {json.dumps(messages, indent=2)}
Existing memories: {json.dumps(memory_dicts, indent=2)}

Decide how to combine the new memories into the database with the existing memories. Use the available tools to ADD, UPDATE, DELETE, or do nothing (NOOP). When finished, use finalize to provide a summary."""

    conversation_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    actions_taken = []
    max_iters = 3

    for iteration in range(max_iters):
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_messages,
            tools=function_definitions,
            tool_choice="auto",
            temperature=1,
            max_tokens=16000,
        )

        message = response.choices[0].message
        conversation_messages.append(message)

        # Check if the model wants to call a function
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "add_memory":
                    memory_text = function_args.get("memory_text", "")
                    categories = function_args.get("categories", [])
                    
                    print("Adding memory: ", memory_text)
                    print("Categories: ", categories)

                    embeddings = await generate_embeddings([memory_text])
                    await insert_memories(
                        memories=[
                            EmbeddedMemory(
                                user_id=user_id,
                                memory_text=memory_text,
                                categories=categories,
                                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                                embedding=embeddings[0],
                            )
                        ]
                    )

                    result = f"Memory: '{memory_text}' was added to DB"
                    actions_taken.append(result)
                    conversation_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                elif function_name == "update":
                    memory_id = function_args.get("memory_id")
                    updated_memory_text = function_args.get("updated_memory_text", "")
                    categories = function_args.get("categories", [])

                    print(
                        "Memory updating: ",
                        "\n Original: ",
                        existing_memories[memory_id].memory_text,
                        "\n New memory text: ",
                        updated_memory_text,
                    )

                    point_id = get_point_id_from_memory_id(memory_id)
                    await delete_records([point_id])

                    embeddings = await generate_embeddings([updated_memory_text])

                    await insert_memories(
                        memories=[
                            EmbeddedMemory(
                                user_id=user_id,
                                memory_text=updated_memory_text,
                                categories=categories,
                                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                                embedding=embeddings[0],
                            )
                        ]
                    )
                    result = f"Memory {memory_id} has been updated to: '{updated_memory_text}'"
                    actions_taken.append(result)
                    conversation_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                elif function_name == "delete":
                    memory_ids_to_delete = function_args.get("memory_ids", [])
                    
                    print("Deleting these memories")
                    point_ids_to_delete = []
                    for memory_id in memory_ids_to_delete:
                        print(existing_memories[memory_id].memory_text)
                        point_ids_to_delete.append(get_point_id_from_memory_id(memory_id))

                    await delete_records(point_ids_to_delete)
                    result = f"Memory {memory_ids_to_delete} deleted"
                    actions_taken.append(result)
                    conversation_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                elif function_name == "noop":
                    result = "No action done"
                    actions_taken.append(result)
                    conversation_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                elif function_name == "finalize":
                    summary = function_args.get("summary", "Completed")
                    return summary
        else:
            conversation_messages.append({
                "role": "user",
                "content": "Please use the finalize function to provide a summary of what you did."
            })

    # If we've exhausted iterations, try to get a final summary
    final_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_messages + [{"role": "user", "content": "Please provide a summary using the finalize function."}],
        tools=function_definitions,
        tool_choice={"type": "function", "function": {"name": "finalize"}},
        temperature=1,
        max_tokens=16000,
    )

    final_message = final_response.choices[0].message
    if final_message.tool_calls:
        for tool_call in final_message.tool_calls:
            if tool_call.function.name == "finalize":
                function_args = json.loads(tool_call.function.arguments)
                return function_args.get("summary", "Completed")

    # Fallback summary
    return "Memory update completed" if actions_taken else "No action taken"


async def update_memories(user_id: int, messages: list[dict]):
    latest_user_message = [x["content"] for x in messages if x["role"] == "user"][-1]
    embedding = (await generate_embeddings([latest_user_message]))[0]

    retrieved_memories = await search_memories(search_vector=embedding, user_id=user_id)

    response = await update_memories_agent(
        user_id=user_id, existing_memories=retrieved_memories, messages=messages
    )
    return response


async def test():
    messages = [{"role": "user", "content": "I want to go Tokyo"}]
    response = await update_memories(user_id=1, messages=messages)

    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())