# Radix-Titan Neural Memory Engine

This document describes the implementation of the Radix-Titan Neural Memory Engine, a next-generation memory system that combines surprisal-based gating, radix path generation, and KV cache stitching for intelligent memory management.

## Architecture Overview

The Radix-Titan architecture replaces the simple "User Prompt → OpenAI → Qdrant → Response" flow with a Logic-First loop:

```
User Prompt → Surprisal Pre-Check → OpenAI Tool Calling → Neural Memory Storage
                                           ↓
KV Cache Stitching ← LMCache-Redis ← Radix Path Generation
```

### Key Components

1. **Surprisal Pre-Check**: Local 1B model determines if a topic is "known" (low surprisal)
2. **Radix-Path Tool Calling**: Generates hierarchical paths (category/topic/detail) for memories
3. **LMCache-Redis Stitching**: Stores and retrieves KV caches for instant context restoration
4. **Recursive MoR Tool**: Traverses knowledge trees for broader context

## Setup Instructions

### 1. Install Dependencies

```bash
# Install with pip
pip install -r requirements.txt

# Or with uv (recommended)
uv pip install -r requirements.txt
```

### 2. Start Required Services

#### Redis (for KV cache storage)
```bash
# Install Redis
brew install redis    # macOS
# or apt-get install redis-server  # Linux

# Start Redis
redis-server
```

#### Qdrant (for vector storage)
```bash
# Using Docker
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 3. Environment Variables

Create a `.env` file with:

```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

# Redis (optional, defaults to redis://localhost:6379)
REDIS_URL=redis://localhost:6379
```

## Usage

### Basic Chat with Neural Memory

```bash
python chatbot.py 1  # Start chat for user ID 1
```

### Testing Components

```bash
python test_neural_memory.py
```

## Architecture Details

### SurprisalGate

Uses a local DialoGPT model to calculate surprisal (negative log probability) for topic novelty detection.

```python
from neural_memory import SurprisalGate

gate = SurprisalGate(surprisal_threshold=2.0)
await gate.initialize()
is_known = await gate.is_topic_known("new topic", "existing context")
```

### TaxonomyLLM

Generates 3-level hierarchical paths for memories.

```python
from neural_memory import TaxonomyLLM

taxonomy = TaxonomyLLM()
await taxonomy.initialize()
path = await taxonomy.generate_radix_path("I work on AI projects at Google")
# Returns: "work/tech/ai" or similar
```

### LMCacheRedisManager

Manages KV cache storage and retrieval with Redis backend.

```python
from neural_memory import LMCacheRedisManager

manager = LMCacheRedisManager()
await manager.initialize()

# Store KV cache
await manager.store_kv_cache("work/project/alpha", kv_cache_data)

# Retrieve KV cache
kv_cache = await manager.retrieve_kv_cache("work/project/alpha")

# Get related paths
parents = await manager.get_parent_paths("work/project/alpha")  # ["work/project", "work"]
siblings = await manager.get_sibling_paths("work/project/alpha")  # Other work/project/* paths
```

### NeuralMemoryEngine

Main orchestrator that combines all components.

```python
from neural_memory import NeuralMemoryEngine
from qdrant_client import AsyncQdrantClient
from openai import AsyncOpenAI

# Initialize
engine = NeuralMemoryEngine()
qdrant = AsyncQdrantClient(url="http://localhost:6333")
openai_client = AsyncOpenAI(api_key="your_key")

await engine.initialize(qdrant, openai_client)

# Store memory with full pipeline
result = await engine.store_memory(
    user_id=1,
    memory_text="User works on machine learning projects",
    categories=["work", "tech"],
    existing_context="Previous conversation about career..."
)

# Prefetch context for inference
context = await engine.prefetch_context("Tell me about my work projects", user_id=1)

# Search recursively
related = await engine.search_recursive_context("work/project/alpha", direction="up")
```

## Integration with Response Generator

The response generator now includes two new tools:

1. **fetch_similar_memories**: Enhanced vector search with radix path metadata
2. **recursive_context_search**: Traverse knowledge trees for broader context

### Example Tool Usage

```python
# In the OpenAI function call
{
    "name": "recursive_context_search",
    "parameters": {
        "radix_path": "work/project/alpha",
        "direction": "up",
        "reason": "Need broader context about the project"
    }
}
```

## Memory Storage Pipeline

1. **Surprisal Check**: Determine if topic is novel
2. **Radix Path Generation**: Create hierarchical path
3. **Qdrant Storage**: Store with metadata including radix_path
4. **KV Cache Storage**: Save to Redis for instant retrieval

## Performance Benefits

- **Reduced API Costs**: Local models for gating and taxonomy
- **Instant Context**: KV cache stitching eliminates re-reading
- **Smart Storage**: Surprisal gating prevents redundant memories
- **Rich Context**: Recursive search provides comprehensive information

## Troubleshooting

### Common Issues

1. **Torch Import Error**: Install PyTorch for your system
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Redis Connection Failed**: Ensure Redis is running
   ```bash
   redis-cli ping  # Should return PONG
   ```

3. **Qdrant Connection Failed**: Check Docker container
   ```bash
   docker ps | grep qdrant
   ```

4. **Model Download Issues**: Models download automatically on first use

### Testing Individual Components

```python
# Test surprisal gate
python -c "
import asyncio
from neural_memory import SurprisalGate
gate = SurprisalGate()
asyncio.run(gate.initialize())
result = asyncio.run(gate.is_topic_known('test'))
print('Surprisal test:', result)
"
```

## Future Enhancements

1. **Real KV Cache Capture**: Integrate with actual model inference to capture real KV caches
2. **Advanced Taxonomy**: Use more sophisticated models for path generation
3. **Memory Consolidation**: Automatically merge related memories
4. **Cross-User Learning**: Share common knowledge patterns across users
5. **Performance Optimization**: GPU acceleration for local models

## File Structure

```
├── neural_memory.py          # Main neural memory engine
├── response_generator.py      # Updated with neural tools
├── test_neural_memory.py      # Component tests
├── requirements.txt           # Dependencies
├── README_NEURAL_MEMORY.md    # This file
└── [existing files...]        # Original codebase
```

## API Reference

### NeuralMemoryEngine

- `initialize(qdrant_client, openai_client)`: Set up all components
- `store_memory(user_id, memory_text, categories, existing_context)`: Store with full pipeline
- `prefetch_context(query, user_id)`: Get context for inference
- `search_recursive_context(radix_path, direction, max_depth)`: Traverse knowledge tree

### Component Classes

- `SurprisalGate`: Topic novelty detection
- `TaxonomyLLM`: Hierarchical path generation
- `LMCacheRedisManager`: KV cache management
- `InferencePrefetcher`: Context prefetching
- `RecursiveContextSearch`: Tree traversal

---

For questions or issues, refer to the component docstrings in `neural_memory.py` or run the test suite with `python test_neural_memory.py`.
