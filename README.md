# Mem2 - AI Memory System

An intelligent AI memory system that maintains conversation context and learns from interactions using vector embeddings and semantic search.

## 🚀 Features

- **Conversational AI**: Interactive chatbot with memory retention
- **Memory Management**: Automatic categorization and storage of conversation memories
- **Semantic Search**: Vector-based retrieval of relevant memories using embeddings
- **OpenAI Integration**: Uses GPT models for response generation and text embeddings
- **Vector Database**: Qdrant for efficient similarity search and storage
- **Async Architecture**: Built with async/await for high performance
- **Rich CLI**: Beautiful console interface with progress indicators

## 🏗️ Architecture

```
User Input → Response Generator → Memory Extraction → Vector DB
      ↓              ↓                    ↓             ↓
   Chatbot ← OpenAI API ← Embedding ← Qdrant Search ← Semantic Retrieval
```

### Core Components

- **Chatbot** (`chatbot.py`): Main entry point for user interactions
- **Response Generator** (`respone_generator.py`): AI agent with ReAct-style reasoning
- **Memory System** (`update_memory.py`): Memory extraction and categorization
- **Embeddings** (`embed.py`): Text-to-vector conversion using OpenAI
- **Vector Database** (`vectordb.py`): Qdrant integration for memory storage and retrieval

## 📋 Prerequisites

- Python 3.10+
- OpenAI API key
- Qdrant vector database (local or cloud)
- uv package manager

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mem2
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

4. **Start Qdrant database**
   ```bash
   # Using Docker
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

   # Or install locally
   # See: https://qdrant.tech/documentation/quickstart/
   ```

## 🚀 Usage

### Running the Chatbot

```bash
# Run with default user ID (1)
uv run python chatbot.py

# Run with specific user ID
uv run python chatbot.py 123
```

### Direct API Usage

```python
import asyncio
from respone_generator import run_chat

# Start chat session for user ID 1
await run_chat(1)
```

### Memory Management

The system automatically:
- Extracts important information from conversations
- Categorizes memories by topic/context
- Stores memories as vector embeddings
- Retrieves relevant memories for future conversations

## ⚙️ Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Vector Database

Default configuration connects to local Qdrant at `http://localhost:6333`. Modify `vectordb.py` for different configurations:

```python
# For cloud Qdrant
client = AsyncQdrantClient(
    url="https://your-cluster.qdrant.cloud",
    api_key="your-api-key"
)
```

### Embedding Dimensions

Currently configured for 64-dimensional embeddings using `text-embedding-3-small`. Modify in `embed.py`:

```python
dimensions=64  # Change as needed
model="text-embedding-3-small"  # Or other embedding models
```

## 📁 Project Structure

```
mem2/
├── chatbot.py              # Main chatbot interface
├── respone_generator.py    # AI response generation with ReAct
├── update_memory.py        # Memory extraction and updates
├── embed.py                # Text embedding generation
├── vectordb.py             # Qdrant vector database operations
├── extraction.py           # Memory extraction utilities
├── main.py                 # Simple entry point
├── pyproject.toml          # Project configuration
├── requirements.txt        # Additional dependencies
├── uv.lock                 # uv lockfile
└── .vscode/                # IDE configuration
    └── settings.json
```

## 🔧 Development

### Adding New Features

1. **New Memory Categories**: Modify categorization logic in `update_memory.py`
2. **Different Embeddings**: Update embedding model in `embed.py`
3. **Custom Tools**: Add new tools to the ReAct agent in `respone_generator.py`

### Testing

```bash
# Run with test data
uv run python embed.py
```

### Code Quality

The project uses:
- **Pydantic**: For data validation and models
- **AsyncIO**: For concurrent operations
- **Type Hints**: Full type annotation coverage
- **Rich**: For beautiful console output

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI**: For GPT models and embeddings
- **Qdrant**: For vector database technology
- **Mem0 AI**: For inspiration on memory systems
- **DSPy**: For programmatic prompting framework

## 🐛 Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Ensure `OPENAI_API_KEY` environment variable is set
   - Check API key validity and billing

2. **"Connection refused" to Qdrant**
   - Verify Qdrant is running on port 6333
   - Check Docker container status

3. **Import errors in IDE**
   - Ensure virtual environment is activated
   - Run `uv sync` to install dependencies

4. **Memory not persisting**
   - Check Qdrant collection creation
   - Verify embedding dimensions match database schema

### Getting Help

- Check the console output for detailed error messages
- Verify all prerequisites are installed
- Test individual components in isolation
