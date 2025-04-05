"""
Quickstart Local - A demo of RAG concepts using simulated embeddings
(no API calls required)
"""

import os
import tempfile
import shutil
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Mock Document class
class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Mock text splitter
class SimpleSplitter:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks based on simple character count"""
        chunks = []
        for doc in documents:
            text = doc.page_content
            
            # Simple character-based chunking
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                if i > 0:
                    start = i
                else:
                    start = 0
                
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]
                
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={**doc.metadata}
                    ))
        
        return chunks

# Mock embedding function
def create_mock_embedding(text: str) -> List[float]:
    """Create a deterministic but simplified mock embedding based on text content"""
    # This is NOT a real embedding, just a simplified simulation
    # We'll use character counts and some basic features as a very simple approximation
    
    # Create a seed from the text to make embeddings deterministic
    random.seed(text)
    
    # Create a mock embedding vector (much smaller than real embeddings)
    embedding = [
        len(text) / 1000,  # Document length
        text.count(' ') / 100,  # Word count approximation
        text.count('.') / 10,  # Sentence count approximation
        text.count('\n') / 5,  # Paragraph count approximation
        random.random(),  # Random noise
        random.random(),  # Random noise
        random.random(),  # Random noise
        random.random()   # Random noise
    ]
    
    # Normalize the embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return embedding

# Mock vector store
class MockVectorStore:
    def __init__(self, documents: List[Document] = None):
        self.documents = documents or []
        self.embeddings = []
        
        if documents:
            for doc in documents:
                self.embeddings.append(create_mock_embedding(doc.page_content))
    
    @classmethod
    def from_documents(cls, documents: List[Document], *args, **kwargs):
        """Create a vector store from documents"""
        return cls(documents)
    
    def save_local(self, folder_path: str, *args, **kwargs):
        """Mock saving the vector store locally"""
        os.makedirs(folder_path, exist_ok=True)
        # In a real implementation, we would serialize documents and embeddings
        print(f"[Mock] Vector store saved to {folder_path}")
    
    @classmethod
    def load_local(cls, folder_path: str, *args, **kwargs):
        """Mock loading the vector store"""
        print(f"[Mock] Vector store loaded from {folder_path}")
        return cls()
    
    def as_retriever(self, search_kwargs=None):
        """Return a retriever that can fetch documents"""
        return MockRetriever(self, search_kwargs or {"k": 2})
    
    def similarity_search(self, query: str, k: int = 2) -> List[Document]:
        """Mock similarity search using simple word overlap"""
        if not self.documents:
            return []
        
        # Create a mock embedding for the query
        query_embedding = create_mock_embedding(query)
        
        # Calculate similarity scores using dot product
        scores = []
        for doc_embedding in self.embeddings:
            score = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            scores.append(score)
        
        # Get the indices of the k highest scores
        if not scores:
            return []
            
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        # Return the corresponding documents
        return [self.documents[i] for i in indices]

# Mock retriever
class MockRetriever:
    def __init__(self, vector_store, search_kwargs):
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs
    
    def invoke(self, query: str) -> List[Document]:
        """Retrieve documents relevant to the query"""
        k = self.search_kwargs.get("k", 2)
        return self.vector_store.similarity_search(query, k=k)

# Mock LLM
class MockLLM:
    def __init__(self, model: str = "mock-model", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
    
    def invoke(self, messages):
        """Generate a response based on provided context and question"""
        # Extract the last message (the prompt with context and query)
        prompt = messages[-1].content if hasattr(messages[-1], 'content') else messages
        
        # Simple rule-based response generation
        if "What is LangGraph?" in prompt:
            return "LangGraph is a library for building stateful, multi-agent applications with LLMs. It extends LangChain with primitives for multi-actor applications, and has features like building graphs with cycles, using external memory/state, and streaming intermediate results."
        
        elif "key components of a RAG system" in prompt:
            return "The key components of a RAG system include: 1) A knowledge base which is a collection of documents or information sources, 2) A retriever component that finds relevant information from the knowledge base, and 3) A generator language model that creates responses based on the retrieved information."
        
        elif "AI agent" in prompt and "work" in prompt:
            return "AI Agents are autonomous systems that use language models to achieve goals. An agent observes its environment, makes decisions, and takes actions. It typically consists of a language model as the core reasoning engine, access to tools and APIs, memory to maintain context, and planning capabilities."
        
        elif "difference between LangGraph and RAG" in prompt:
            return "LangGraph is a library for building stateful multi-agent applications, while RAG (Retrieval Augmented Generation) is a technique that enhances LLM outputs by incorporating information retrieved from a knowledge base. LangGraph can be used to implement complex workflows including RAG systems, while RAG itself is a specific approach to improving LLM responses with external knowledge."
        
        else:
            return "I don't have enough information to answer this question based on the provided context."

# Create a temporary directory for our sample knowledge base
def create_sample_knowledge_base():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    # Sample content
    samples = [
        {
            "filename": "langgraph.txt",
            "content": """LangGraph is a library for building stateful, multi-agent applications with LLMs.
            
LangGraph extends LangChain with the primitives needed to build stateful, multi-actor applications. 
It helps manage state and coordinate the exchange of messages between different components in an application, 
like a conversation between multiple agents or a workflow with multiple steps.

Features:
- Building graphs with cycles
- Using external memory/state
- Streaming intermediate results
- Compiler optimizations
- Types and type checking
- Human-in-the-loop capabilities
"""
        },
        {
            "filename": "rag.txt",
            "content": """Retrieval Augmented Generation (RAG) is a technique that enhances LLM outputs 
by incorporating relevant information retrieved from a knowledge base.

RAG systems combine the strengths of retrieval-based and generation-based approaches. 
They retrieve relevant information from a knowledge base and then use this information 
to generate more accurate and informed responses.

Key components of a RAG system:
1. Knowledge base - A collection of documents or information sources
2. Retriever - A component that finds relevant information from the knowledge base
3. Generator - A language model that creates responses based on the retrieved information

Benefits of RAG:
- More accurate and factual responses
- Reduced hallucinations
- Access to domain-specific knowledge
- Ability to cite sources
"""
        },
        {
            "filename": "agents.txt",
            "content": """AI Agents are autonomous systems that use language models to achieve goals.

An agent can observe its environment, make decisions, and take actions to achieve its goals.
It typically consists of:
1. A language model as the core reasoning engine
2. Access to tools and APIs to interact with the world
3. Memory to maintain context and learn from past experiences
4. Planning capabilities to break down complex tasks

Types of agents:
- ReAct agents: Reason and act in an alternating manner
- Reflection agents: Self-critique and improve their own outputs
- Tool-using agents: Utilize external tools to accomplish tasks
- Multi-agent systems: Multiple agents collaborating to solve problems
"""
        }
    ]
    
    # Write the samples to files
    for sample in samples:
        file_path = os.path.join(temp_dir, sample["filename"])
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(sample["content"])
        print(f"Created sample file: {file_path}")
    
    return temp_dir

# Process the knowledge base
def process_knowledge_base(kb_dir):
    # Load documents
    documents = []
    for file_path in Path(kb_dir).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            doc = Document(page_content=text, metadata={"source": file_path.name})
            documents.append(doc)
    
    print(f"Loaded {len(documents)} documents")
    
    # Split documents
    text_splitter = SimpleSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    print(f"Split into {len(splits)} chunks")
    
    # Create mock vector store
    vector_store = MockVectorStore.from_documents(splits)
    
    # Save the vector store
    persist_dir = "./quickstart_kb_local"
    os.makedirs(persist_dir, exist_ok=True)
    vector_store.save_local(persist_dir)
    
    print(f"Created mock vector store in {persist_dir}")
    
    return persist_dir, splits

# Create a simple RAG chain
def create_rag_chain(persist_dir, documents=None):
    """
    Create a simple RAG chain with mock components
    
    Args:
        persist_dir: Directory where the vector store is located
        documents: Optional list of documents to use instead of loading from disk
    """
    # Initialize mock vector store
    if documents:
        vector_store = MockVectorStore(documents)
    else:
        vector_store = MockVectorStore.load_local(persist_dir)
    
    # Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Initialize mock LLM
    llm = MockLLM(model="mock-model", temperature=0)
    
    # Create a mock chain that takes a question and returns an answer
    class MockChain:
        def __init__(self, retriever, llm):
            self.retriever = retriever
            self.llm = llm
        
        def invoke(self, query):
            # Retrieve documents
            docs = self.retriever.invoke(query)
            
            # Format context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            prompt = f"""Answer the question based ONLY on the following context:

{context}

Question: {query}

If the answer is not contained within the context, say "I don't have enough information to answer this question."
"""
            
            # Generate answer
            answer = self.llm.invoke(prompt)
            
            return answer
    
    return MockChain(retriever, llm)

# Run demo queries
def run_demo_queries(chain):
    queries = [
        "What is LangGraph?",
        "What are the key components of a RAG system?",
        "How does an AI agent work?",
        "What is the difference between LangGraph and RAG?",
    ]
    
    for query in queries:
        print("\n" + "=" * 50)
        print(f"Question: {query}")
        print("-" * 50)
        answer = chain.invoke(query)
        print(f"Answer: {answer}")

# Main function
def main():
    print("Knowledge Base Quickstart Demo (Local Version)")
    print("=" * 40)
    print("This demo uses simulated components with no API calls.")
    
    try:
        # Create sample knowledge base
        kb_dir = create_sample_knowledge_base()
        
        # Process the knowledge base
        persist_dir, documents = process_knowledge_base(kb_dir)
        
        # Create a RAG chain and run demo queries
        chain = create_rag_chain(persist_dir, documents)
        
        # Run the demo queries
        run_demo_queries(chain)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if 'kb_dir' in locals():
            shutil.rmtree(kb_dir)
            print(f"\nCleaned up temporary directory: {kb_dir}")
        
        print("\nDemo completed!")

if __name__ == "__main__":
    main() 