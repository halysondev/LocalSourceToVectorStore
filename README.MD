# Source Code Indexing System

This project implements a powerful source code indexing system that processes, embeds, and stores code in a vector database for efficient semantic search. It supports multiple programming languages and various encoding formats.

## Features

- Multi-language support with automatic language detection
- Binary file detection and filtering
- Multiple encoding support (gb18030, gb2312, utf-8, utf-16, latin-1)
- Parallel processing capabilities
- Support for both OpenAI and HuggingFace embeddings
- Vector store options: Qdrant or Chroma
- Progress tracking with tqdm
- Git repository cloning support

## Technologies Used

### AI and Machine Learning
- **OpenAI Embeddings**: Production-ready embeddings for high-quality semantic search (optional)
- **HuggingFace Transformers**: Open-source alternative for text embeddings using the MiniLM-L6-v2 model
- **sentence-transformers**: Framework for state-of-the-art text embeddings

### Vector Databases
- **Qdrant**: High-performance vector similarity search engine with:
  - GRPC support for efficient communication
  - Cosine similarity metrics
  - Collection-based organization
  - Real-time indexing
- **Chroma**: Lightweight alternative vector database with:
  - Local persistence
  - Simple API
  - Embedded operation

### Code Processing
- **LangChain**: Framework for:
  - Document loading and processing
  - Text splitting and chunking
  - Language detection
  - Integration with various embedding providers
- **GitPython**: Git repository management and cloning
- **chardet**: Automatic character encoding detection

### Performance and Utilities
- **tqdm**: Progress tracking and monitoring
- **concurrent.futures**: Parallel processing implementation
- **CUDA**: GPU acceleration support for embeddings generation

## Configuration Options

### Basic Settings
```python
FROM_GIT = False           # Enable to clone from a Git repository
GIT_URL = ""              # Git repository URL to clone
repo_path = "source"   # Local path for processing

USE_OPENAI = False        # Toggle between OpenAI and HuggingFace embeddings
```

### Embedding Configuration
```python
# OpenAI Settings
OPENAI_API_KEY = "sk-xxx-YourKeyHere"    # Required if USE_OPENAI is True

# HuggingFace Settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
```

### Text Processing Settings
```python
CHUNK_SIZE = 500          # Size of text chunks for processing
CHUNK_OVERLAP = 200       # Overlap between chunks
PARSER_THRESHOLD = 1000   # Threshold for the language parser
```

### Vector Store Settings
```python
VECTOR_CHROMA = False     # Use Chroma as vector store
VECTOR_QDRANT = True      # Use Qdrant as vector store
collection_name = "source"
qdrant_url = "http://127.0.0.1:6333"
```

### Performance Settings
```python
BATCH_SIZE = 250         # Number of documents per batch
MAX_WORKERS = 1          # Number of parallel workers
DEVICE_TYPE = "cuda"     # Device for HuggingFace embeddings
```

## File Processing

### File Filtering
The system automatically filters files based on:
- Excluded file extensions (binaries, executables, etc.)
- Binary content detection
- Text file whitelist

### Encoding Support
Supports multiple encodings with fallback mechanisms:
1. Tries predefined encodings (gb18030, gb2312, utf-8, utf-16, latin-1)
2. Uses chardet for automatic encoding detection
3. Implements error handling for undecodable files

## Language Support

Automatically detects and processes multiple programming languages:
- Python
- C/C++
- JavaScript
- Java
- Lua
- PHP
- Ruby
- Rust
- Swift
- TypeScript
- HTML
- Perl
- Markdown (default for unknown extensions)

## Vector Store Integration

### Qdrant
- Default vector store option
- Uses cosine similarity for vector comparison
- Supports collection management
- GRPC communication option

### Chroma
- Alternative vector store option
- Supports local persistence
- Collection-based organization

## Processing Pipeline

1. File Loading
   - Scans target directory
   - Filters invalid/binary files
   - Prepares valid files for processing

2. Document Processing
   - Detects file language
   - Handles multiple encodings
   - Splits documents into manageable chunks

3. Embedding Generation
   - Creates embeddings using chosen provider
   - Processes in batches for efficiency

4. Vector Store Integration
   - Stores embeddings in chosen vector database
   - Maintains metadata and relationships

## Requirements

The system requires several Python packages:
- langchain
- qdrant-client
- chromadb (optional)
- openai (optional)
- sentence-transformers
- tqdm
- GitPython
- chardet

## Usage

1. Configure the settings in the script according to your needs
2. Ensure all required dependencies are installed
3. Run the script to process your codebase
4. Use the resulting vector store for semantic code search

## Performance Considerations

- Batch processing helps manage memory usage
- Parallel processing can be enabled by increasing MAX_WORKERS
- CUDA support available for HuggingFace embeddings
- Progress tracking provides visibility into processing status
