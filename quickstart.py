"""
Quickstart script to demonstrate the knowledge base import and query process
with a small sample dataset.
"""

import os
import tempfile
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Setup API key
def setup_api_key():
    # Use real API key
    if not os.environ.get("OPENAI_API_KEY"):
        api_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
        print("API key set for this session.")

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    print(f"Split into {len(splits)} chunks")
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    
    # Save the vector store
    persist_dir = "./quickstart_kb"
    os.makedirs(persist_dir, exist_ok=True)
    vector_store.save_local(persist_dir, index_name="index")
    
    print(f"Created vector store in {persist_dir}")
    
    return persist_dir

# Create a simple RAG chain
def create_rag_chain(persist_dir):
    # Initialize models
    embeddings = OpenAIEmbeddings()
    
    # For demo purposes - use a simple model with lower cost
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Load the vector store
    vector_store = FAISS.load_local(
        folder_path=persist_dir,
        embeddings=embeddings,
        index_name="index"
    )
    
    # Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Create a prompt template
    template = """Answer the question based ONLY on the following context:

{context}

Question: {question}

If the answer is not contained within the context, say "I don't have enough information to answer this question."
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the RAG chain
    chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Demo queries
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
    print("Knowledge Base Quickstart Demo")
    print("=" * 30)
    
    # Setup API key
    setup_api_key()
    
    try:
        # Create sample knowledge base
        kb_dir = create_sample_knowledge_base()
        
        # Process the knowledge base
        persist_dir = process_knowledge_base(kb_dir)
        
        # Create a RAG chain and run demo queries
        chain = create_rag_chain(persist_dir)
        
        # Run the demo queries with the actual API key
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