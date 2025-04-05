import os
import argparse
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Default directory where the vector store is located
DEFAULT_PERSIST_DIRECTORY = "./knowledge_base_db"

def setup_api_keys():
    """Load API keys from .env file and set up necessary API keys."""
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"DEBUG: Value of OPENAI_API_KEY from os.environ (after load_dotenv): {'Exists and is hidden' if api_key and api_key != 'YOUR_API_KEY_HERE_REPLACE_THIS' else 'Not found, empty, or placeholder'}")
    
    if not api_key or api_key == "YOUR_API_KEY_HERE_REPLACE_THIS":
        print("Error: OPENAI_API_KEY not found in environment or .env file, or it's still the placeholder.")
        print("Please ensure you have created a .env file with your actual key (e.g., OPENAI_API_KEY=sk-...).")
        raise ValueError("Valid OpenAI API key is required.")
    
    # Optional: Add a check for likely invalid keys (e.g., too short)
    if len(api_key) < 40:
         print(f"Warning: API key seems short ({len(api_key)} chars). Ensure it's correct.")
         
    print("Using OpenAI API key loaded from environment/.env file.")

def create_rag_chain(persist_directory: str = DEFAULT_PERSIST_DIRECTORY):
    """
    Create and return a RAG chain that retrieves documents and generates answers.
    
    Args:
        persist_directory: Directory where the vector store is located
        
    Returns:
        A chain that takes a question and returns an answer
    """
    # Initialize embedding model
    embeddings = OpenAIEmbeddings()
    
    # Load the vector store
    print(f"Loading vector store from {persist_directory}...")
    try:
        vectorstore = FAISS.load_local(
            folder_path=persist_directory,
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        # Print diagnostic info
        print(f"Vector store loaded. Index contains data.")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        raise
    
    # Create a retriever with improved search parameters
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Use standard similarity search instead of threshold
        search_kwargs={
            "k": 5,  # Adjusted number of documents
        }
    )
    
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create a prompt template with more flexible instructions
    template = """You are a helpful AI assistant that answers questions based on the provided knowledge base.

Given the following context information from the knowledge base:

{context}

Answer the question: {question}

Guidelines for your response:
1. Primarily base your answer on the provided context information.
2. You may use your general knowledge to interpret and explain concepts found in the context, but not to introduce completely new information.
3. If the question isn't directly addressed in the context but you can infer a reasonable answer, provide one while noting it's based on interpretation.
4. If the context doesn't provide enough information for even a reasonable inference, explain what information is missing and suggest related questions that might be better answered with the available knowledge.
5. Include relevant quotes or specific references from the context when helpful.
6. Keep your answer comprehensive but concise.
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

def query_knowledge_base(
    query: str,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    show_sources: bool = False
):
    """
    Query the knowledge base and get an answer.
    
    Args:
        query: Question to ask
        persist_directory: Directory where the vector store is located
        show_sources: Whether to show the source documents
    """
    print(f"\nQuestion: {query}")
    print("-" * 50)
    
    # Create the RAG chain
    chain = create_rag_chain(persist_directory)
    
    # Get the answer
    answer = chain.invoke(query)
    
    print(f"Answer: {answer}")
    
    # Optionally show source documents
    if show_sources:
        print("\nSources:")
        print("-" * 50)
        
        # Initialize embedding model and load the vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            folder_path=persist_directory,
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        
        # Get the source documents with scores
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=5) # Adjusted to match retriever
        
        if not docs_with_scores:
            print("No relevant sources found.")
        else:
            for i, (doc, score) in enumerate(docs_with_scores):
                relevance = score if isinstance(score, float) else float(score)
                relevance_percentage = min(100, max(0, 100 * (1 - relevance)))
                
                print(f"Source {i+1}: {doc.metadata.get('source', 'unknown')}")
                print(f"Relevance: {relevance_percentage:.1f}%")
                print(f"Content: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else f"Content: {doc.page_content}")
                print("-" * 50)

def interactive_mode(persist_directory: str = DEFAULT_PERSIST_DIRECTORY, show_sources: bool = False):
    """
    Run an interactive query session.
    
    Args:
        persist_directory: Directory where the vector store is located
        show_sources: Whether to show the source documents
    """
    print("Knowledge Base Query Interface")
    print("------------------------------")
    print("Type 'exit', 'quit', or 'q' to end the session\n")
    
    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        query_knowledge_base(query, persist_directory, show_sources)
        print("\n")

def check_knowledge_base(persist_directory: str = DEFAULT_PERSIST_DIRECTORY):
    """
    Check the content of the knowledge base and display statistics.
    
    Args:
        persist_directory: Directory where the vector store is located
    """
    print(f"Checking knowledge base in {persist_directory}...")
    
    try:
        # Initialize embedding model
        print("Initializing OpenAIEmbeddings...")
        try:
            embeddings = OpenAIEmbeddings()
            print("OpenAIEmbeddings initialized successfully.")
        except Exception as e:
            print(f"Error initializing OpenAIEmbeddings: {e}")
            raise
        
        # Load the vector store
        print(f"Loading vector store from {persist_directory}...")
        vectorstore = FAISS.load_local(
            folder_path=persist_directory,
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
        
        # Sample documents using diverse queries
        print("\nSampling documents from knowledge base...")
        queries = ["overview", "process", "rules", "definition", "guidelines"]
        sampled_docs = {}
        
        for query in queries:
            print(f"  Running similarity search for query: '{query}'")
            try:
                docs = vectorstore.similarity_search(query, k=3)
                for doc in docs:
                    if doc.page_content not in sampled_docs:
                        sampled_docs[doc.page_content] = doc
                print(f"    Found {len(docs)} results for '{query}'.")
            except Exception as e:
                print(f"    Error during similarity search for '{query}': {e}")
                # Optionally decide whether to continue or raise
                # continue
                raise
        
        # Display the unique sampled documents
        print("\nSampled documents:")
        print("-" * 50)
        if not sampled_docs:
            print("Could not retrieve sample documents.")
        else:
            for i, doc in enumerate(list(sampled_docs.values())[:10]): # Show up to 10 unique samples
                print(f"Document {i+1}:")
                print(f"Source: {doc.metadata.get('source', 'unknown')}")
                print(f"Content: {doc.page_content[:150]}..." if len(doc.page_content) > 150 else f"Content: {doc.page_content}")
                print("-" * 50)
        
        print("Knowledge base check completed.")
        
    except Exception as e:
        print(f"Error checking knowledge base: {e}")
        # Avoid raising here to see final output if possible
        # raise

def main():
    parser = argparse.ArgumentParser(description="Query a knowledge base")
    
    # Arguments
    parser.add_argument("--query", type=str, 
                        help="Question to ask (if not provided, interactive mode is started)")
    parser.add_argument("--persist-dir", type=str, default=DEFAULT_PERSIST_DIRECTORY, 
                        help=f"Directory where the vector store is located (default: {DEFAULT_PERSIST_DIRECTORY})")
    parser.add_argument("--show-sources", action="store_true", 
                        help="Show source documents for the answer")
    parser.add_argument("--check-kb", action="store_true",
                        help="Check the content of the knowledge base")
    
    args = parser.parse_args()
    
    # Set up API keys
    setup_api_keys()
    
    if args.check_kb:
        check_knowledge_base(args.persist_dir)
        return
    
    if args.query:
        query_knowledge_base(args.query, args.persist_dir, args.show_sources)
    else:
        interactive_mode(args.persist_dir, args.show_sources)

if __name__ == "__main__":
    main() 