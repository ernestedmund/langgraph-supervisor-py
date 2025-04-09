# Knowledge Base Import and Query

This project provides tools to import an existing knowledge base into a vector store for use with RAG (Retrieval-Augmented Generation) applications.

## Features

- Import documents from multiple file formats (TXT, PDF, CSV, Markdown, HTML, JSON)
- Split documents into manageable chunks for effective retrieval
- Create embeddings and store them in a FAISS vector database
- Query the knowledge base with natural language questions
- Interactive query interface with source citations

## Prerequisites

- Python 3.10 or newer
- OpenAI API key (for embeddings and language model)

## Installation

1. Install required packages:

```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu tqdm bs4
```

2. Clone or download this repository

## Usage

### Importing Your Knowledge Base

Use the `import_knowledge_base.py` script to import your knowledge base:

```bash
python import_knowledge_base.py --source /path/to/your/knowledge_base
```

#### Options

- `--source`: Path to your knowledge base directory (required)
- `--file-types`: Comma-separated list of file types to process (default: "txt,pdf,csv,md,html,json")
- `--chunk-size`: Size of document chunks (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)
- `--persist-dir`: Directory to persist vector store (default: "./knowledge_base_db")
- `--batch-size`: Batch size for processing documents (default: 100)
- `--query`: Test query to run after importing (optional)
- `--verbose/--quiet`: Control output verbosity

#### Processing Steps

You can run specific steps independently:

- `--load-only`: Only load documents, don't split or create vector store
- `--split-only`: Only split documents, don't create vector store
- `--query-only`: Only run a query, don't load, split, or create vector store (requires `--query`)

#### Examples

Import all supported files from a directory:
```bash
python import_knowledge_base.py --source /path/to/your/knowledge_base
```

Import only PDF and TXT files with custom chunk size:
```bash
python import_knowledge_base.py --source /path/to/your/knowledge_base --file-types pdf,txt --chunk-size 500
```

Test a query against an existing knowledge base:
```bash
python import_knowledge_base.py --source /path/to/your/knowledge_base --query-only --query "What is RAG?"
```

### Querying Your Knowledge Base

Once your knowledge base is imported, use the `query_knowledge_base.py` script to query it:

```bash
python query_knowledge_base.py
```

This will start an interactive session where you can ask questions.

#### Options

- `--query`: Question to ask (if not provided, interactive mode is started)
- `--persist-dir`: Directory where the vector store is located (default: "./knowledge_base_db")
- `--show-sources`: Show source documents for the answer

#### Examples

Ask a single question:
```bash
python query_knowledge_base.py --query "What is LangGraph?" --show-sources
```

Start interactive mode with source citations:
```bash
python query_knowledge_base.py --show-sources
```

## Multi-Agent Supervisor Architecture

This project now includes a sophisticated multi-agent architecture powered by LangGraph, allowing for domain-specific knowledge retrieval and conversational memory.

### Key Features

- **Supervisor Agent**: Routes queries to the appropriate specialist agent based on topic and conversation context
- **Consolidated Domain Agents**: Specialized agents with access to domain-specific knowledge bases
- **Conversational Memory**: Maintains context across conversation turns, enabling follow-up questions

### Available Agents

- **RealEstateAgent**: Handles questions about general real property, assessments, mapping, etc.
- **OwnershipTransferAgent**: Specializes in property transfers, Prop 58/193, Prop 19, etc.
- **BusinessPersonalAgent**: Focuses on business personal property, boats, aircraft, etc.
- **ExemptionsAppealsAgent**: Addresses exemptions and assessment appeals processes

### Usage

1. First, ensure you've imported your knowledge base with the consolidated agent structure:

```bash
python import_knowledge_base.py --source /path/to/your/knowledge_base
```

2. Run the supervisor-based interactive interface:

```bash
python supervisor_main.py
```

3. Ask questions naturally. The system will:
   - Route your question to the most relevant agent
   - Allow follow-up questions that reference previous context
   - Maintain conversation history for more accurate responses

### How It Works

1. The `supervisor_main.py` script defines a LangGraph workflow where:
   - A supervisor node analyzes incoming queries and routes them to specialist agents
   - Specialist agents retrieve information from their domain-specific vector stores
   - All messages are accumulated in the conversation history

2. The `agent_tools.py` script provides:
   - Tools for creating RAG chains customized for each domain
   - Conversational memory management functions
   - Question condensation for handling follow-up questions

### Customization

You can customize various aspects of the supervisor architecture:

- Adjust routing behavior in the `create_supervisor_node()` function
- Modify agent prompts in the `create_specialist_rag_chain()` function
- Change the number of retrieved documents by adjusting the `k_value` parameter
- Add new specialist agents by updating the `agent_sources` dictionary in `import_knowledge_base.py`

## Customization

### Handling Different File Types

For specialized document types, you may need to customize the import process:

1. For JSON files: Adjust the `jq` parameter in `import_knowledge_base.py` to match your JSON structure
2. For CSV files: You may need to specify column mappings
3. For structured data: Consider creating custom document loaders

### Adjusting RAG Parameters

To customize retrieval and generation:

1. Modify the retriever in `create_rag_chain()` in `query_knowledge_base.py`
2. Adjust the prompt template to better suit your specific knowledge domain
3. Change the model or its parameters (e.g., temperature) to control response style

## Alternative Vector Stores

The current implementation uses FAISS (Facebook AI Similarity Search) for the vector store. If you prefer other options:

1. For Chroma: Install with `pip install chromadb` (requires Visual C++ Build Tools on Windows)
2. For Pinecone: Install with `pip install pinecone-client`
3. For Weaviate: Install with `pip install weaviate-client`

Then modify the import and query scripts to use your preferred vector store.

## Troubleshooting

### Common Issues

- **"No module found" errors**: Make sure all dependencies are installed
- **Memory errors**: For large knowledge bases, try increasing batch size or running with smaller chunks
- **Rate limiting**: Adjust sleep times in `create_vector_store()` if you hit API rate limits
- **Poor query results**: Try adjusting the chunk size/overlap or modifying the RAG prompt

## License

This project is licensed under the MIT License - see the LICENSE file for details. 