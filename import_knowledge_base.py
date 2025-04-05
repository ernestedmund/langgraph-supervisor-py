import os
import time
from typing import List, Optional
import argparse
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configure available document loaders
from langchain_community.document_loaders import (
    DirectoryLoader,
    # PyPDFLoader, # Commented out
    CSVLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
    UnstructuredPDFLoader # Added UnstructuredPDFLoader
)

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

def setup_api_keys():
    """Load API keys from .env file."""
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"DEBUG (Import Script): Value of OPENAI_API_KEY from os.environ: {'Exists and is hidden' if api_key and api_key != 'YOUR_API_KEY_HERE_REPLACE_THIS' else 'Not found, empty, or placeholder'}")
    if not api_key or api_key == "YOUR_API_KEY_HERE_REPLACE_THIS":
        print("Error: OPENAI_API_KEY not found in environment or .env file, or it's still the placeholder.")
        print("Please ensure you have created a .env file with your actual key (e.g., OPENAI_API_KEY=sk-...).")
        raise ValueError("Valid OpenAI API key is required.")
    if len(api_key) < 40:
         print(f"Warning: API key seems short ({len(api_key)} chars). Ensure it's correct.")
    print("Using OpenAI API key loaded from environment/.env file for import.")

def load_documents(
    source_dir: str, 
    file_types: List[str] = ["txt", "pdf", "csv", "md", "html", "json"],
    verbose: bool = True
) -> List[Document]:
    """
    Load documents from a directory based on specified file types.
    
    Args:
        source_dir: Directory containing the knowledge base files
        file_types: List of file extensions to process
        verbose: Whether to print progress information
    
    Returns:
        List of Document objects
    """
    if verbose:
        print(f"Loading documents from {source_dir}...")
    
    documents = []
    
    # Define loaders for each file type
    loaders = []
    
    if "txt" in file_types:
        if verbose:
            print("Adding TXT loader...")
        loaders.append(DirectoryLoader(
            source_dir, 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            show_progress=verbose
        ))
        
    # --- Modified PDF Loading --- 
    if "pdf" in file_types:
        if verbose:
            print("Adding UnstructuredPDF loader...")
        # UnstructuredPDFLoader can often handle directories, but iterating might give more control/logging
        pdf_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        
        for pdf_file in tqdm(pdf_files, disable=not verbose):
            try:
                # Using UnstructuredPDFLoader with default settings
                # Can add mode="elements" or other strategies if needed
                loader = UnstructuredPDFLoader(pdf_file, mode="single", strategy="fast")
                loaded_docs = loader.load()
                # Add source manually as Unstructured might not always add it consistently
                for doc in loaded_docs:
                    doc.metadata['source'] = os.path.basename(pdf_file)
                documents.extend(loaded_docs)
                if verbose and len(loaded_docs) == 0:
                    print(f"Warning: No content extracted from PDF {pdf_file}")
            except Exception as e:
                print(f"Error loading PDF {pdf_file} with UnstructuredPDFLoader: {e}")
    # --- End Modified PDF Loading --- 
    
    if "csv" in file_types:
        if verbose:
            print("Adding CSV loader...")
        # Note: Ensure CSVLoader args are appropriate for your CSV structure if needed
        loaders.append(DirectoryLoader(
            source_dir, 
            glob="**/*.csv", 
            loader_cls=CSVLoader, 
            show_progress=verbose
        ))
    
    if "md" in file_types:
        if verbose:
            print("Adding Markdown loader...")
        loaders.append(DirectoryLoader(
            source_dir, 
            glob="**/*.md", 
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=verbose
        ))
    
    if "html" in file_types:
        if verbose:
            print("Adding HTML loader...")
        loaders.append(DirectoryLoader(
            source_dir, 
            glob="**/*.html", 
            loader_cls=UnstructuredHTMLLoader,
            show_progress=verbose
        ))

    if "json" in file_types:
        if verbose:
            print("Adding JSON loader...")
        # This simple JSON loading might need customization based on your JSON structure
        json_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(".json"):
                    json_files.append(os.path.join(root, file))
        
        for json_file in tqdm(json_files, disable=not verbose):
            try:
                loader = JSONLoader(file_path=json_file, jq='.', text_content=False) # Assuming content is structured
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading JSON {json_file}: {e}")
    
    # Load documents using the remaining DirectoryLoaders
    for loader in loaders:
        try:
            docs = loader.load()
            documents.extend(docs)
            if verbose:
                print(f"Loaded {len(docs)} documents from {loader.__class__.__name__}")
        except Exception as e:
            print(f"Error with loader {loader.__class__.__name__}: {e}")
    
    if verbose:
        print(f"Loaded {len(documents)} total documents")
    
    return documents

def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    verbose: bool = True
) -> List[Document]:
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        verbose: Whether to print progress information
    
    Returns:
        List of split Document objects
    """
    if verbose:
        print(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    
    if verbose:
        print(f"Split into {len(splits)} chunks")
    
    return splits

def create_vector_store(
    documents: List[Document],
    persist_directory: str,
    batch_size: int = 100,
    verbose: bool = True
) -> None:
    """
    Create embeddings and store in a vector database.
    
    Args:
        documents: List of document chunks to embed
        persist_directory: Directory to store the vector database
        batch_size: Number of documents to process at once
        verbose: Whether to print progress information
    """
    if verbose:
        print(f"Creating embeddings and vector store in {persist_directory}...")
    
    # Initialize embedding model
    embeddings = OpenAIEmbeddings()
    
    # Create directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # FAISS index path
    index_path = os.path.join(persist_directory, "index")
    
    # Check if the vector store already exists
    if os.path.exists(index_path + ".faiss"):
        if verbose:
            print(f"Loading existing vector store from {index_path}")
        
        # Load existing index
        vector_store = FAISS.load_local(
            folder_path=persist_directory,
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        
        # Process documents in batches
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(documents), batch_size), total=total_batches, disable=not verbose):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
            
            # Optional: Sleep to avoid rate limiting
            if i + batch_size < len(documents):
                time.sleep(0.5)
            
            # Save after each batch
            vector_store.save_local(persist_directory, index_name="index")
    else:
        if verbose:
            print(f"Creating new vector store in {persist_directory}")
        
        if documents:
            # For a new vector store, we'll create it with the first batch
            # and then add the rest
            first_batch = documents[:min(batch_size, len(documents))]
            rest = documents[min(batch_size, len(documents)):]
            
            # Create with first batch
            vector_store = FAISS.from_documents(
                documents=first_batch,
                embedding=embeddings
            )
            
            # Save the initial index
            vector_store.save_local(persist_directory, index_name="index")
            
            # Add the rest in batches
            total_batches = (len(rest) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(rest), batch_size), total=total_batches, disable=not verbose):
                batch = rest[i:i + batch_size]
                vector_store.add_documents(batch)
                
                # Save after each batch
                vector_store.save_local(persist_directory, index_name="index")
                
                # Optional: Sleep to avoid rate limiting
                if i + batch_size < len(rest):
                    time.sleep(0.5)
        else:
            if verbose:
                print("No documents to process")
            return
    
    if verbose:
        print(f"Successfully created vector store with {len(documents)} documents")

def main():
    parser = argparse.ArgumentParser(description="Import knowledge base domains into consolidated vector stores")
    
    parser.add_argument("--source", type=str, required=True, 
                        help="Base directory containing domain subdirectories")
    parser.add_argument("--db-prefix", type=str, default="./db", 
                        help="Prefix for the output database directories (default: './db')")
    parser.add_argument("--common-folders", type=str, default="general", 
                        help="Comma-separated list of folder names to treat as common knowledge (will be included in ALL agent databases)")
    parser.add_argument("--file-types", type=str, default="txt,pdf,csv,md,html,json", 
                        help="Comma-separated list of file types to process")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, 
                        help=f"Size of document chunks")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, 
                        help=f"Overlap between chunks")
    parser.add_argument("--batch-size", type=int, default=100, 
                        help="Batch size for processing documents")
    parser.add_argument("--verbose", action="store_true", default=True, 
                        help="Print detailed progress information")
    parser.add_argument("--quiet", action="store_false", dest="verbose", 
                        help="Minimize output")

    args = parser.parse_args()
    
    setup_api_keys()
    file_types = [ft.strip() for ft in args.file_types.split(",")]
    common_folder_names = {name.strip() for name in args.common_folders.split(",") if name.strip()}
    
    base_source_dir = Path(args.source)
    if not base_source_dir.is_dir():
        print(f"Error: Source directory '{args.source}' not found.")
        return

    # --- Define Agent Groupings and their Source Folders --- 
    # Keys are the desired agent names (will also be part of the DB name)
    # Values are LISTS of source subdirectory names within args.source
    agent_sources = {
        "RealEstateAgent": [
            "residential", "commercial", "manufactured_housing", 
            "mapping", "calamities", "restricted", "general", "FAQ" # Add other relevant folders
        ],
        "OwnershipTransferAgent": [
            "change_in_ownership", "general", "FAQ" # Add others like parent-child, trusts etc. if separate
        ],
        "BusinessPersonalAgent": [
            "business_properties" "general", "FAQ"  # Add others like boats, aircraft if separate
        ],
        "ExemptionsAppealsAgent": [
            "exemptions", "assessment_appeals", "general", "FAQ"# Add others like homeowner, veteran etc. if separate
        ]
        # Add a General/Other agent if needed, separate from common folders
        # "GeneralQueriesAgent": ["general"] # Example if general isn't common
    }
    # --- End Agent Groupings ---

    print(f"Base source directory: {base_source_dir}")
    print(f"Common knowledge folders: {common_folder_names or 'None'}")

    # Process each defined agent group
    for agent_name, specific_source_names in agent_sources.items():
        persist_dir = f"{args.db_prefix}_{agent_name}"
        all_source_paths_for_agent = []

        print("\n" + "="*50)
        print(f"Processing Agent Group: {agent_name}")
        print(f"Output DB: {persist_dir}")

        # 1. Collect all source directory paths for this agent (specific + common)
        source_names_for_agent = set(specific_source_names) | common_folder_names
        print(f" -> Including source folders: {source_names_for_agent}")
        
        missing_folders = []
        for folder_name in source_names_for_agent:
            source_path = base_source_dir / folder_name
            if source_path.is_dir():
                all_source_paths_for_agent.append(source_path)
            else:
                missing_folders.append(str(source_path))
        
        if missing_folders:
            print(f"Warning: The following source folders were not found and will be skipped for agent {agent_name}: {missing_folders}")
        
        if not all_source_paths_for_agent:
            print(f"Error: No valid source folders found for agent {agent_name}. Skipping.")
            continue
        
        # 2. Load documents from all collected paths
        all_documents_for_agent = []
        print(f" -> Loading documents from {len(all_source_paths_for_agent)} folder(s)...")
        for source_path in all_source_paths_for_agent:
            print(f"    - Loading from: {source_path}")
            try:
                docs = load_documents(str(source_path), file_types, verbose=args.verbose)
                all_documents_for_agent.extend(docs)
            except Exception as e:
                 print(f"    - Error loading from {source_path}: {e}")
                 # Decide whether to continue or stop for this agent
                 # raise e # Stop processing this agent
                 continue # Skip this folder

        if not all_documents_for_agent:
            print(f"No documents successfully loaded for agent {agent_name}. Skipping DB creation.")
            continue
            
        print(f" -> Total documents loaded for {agent_name}: {len(all_documents_for_agent)}")

        # 3. Split combined documents
        splits = split_documents(all_documents_for_agent, args.chunk_size, args.chunk_overlap, verbose=args.verbose)
        if not splits:
            print(f"No splits generated for agent {agent_name}. Skipping DB creation.")
            continue
        
        # 4. Create/update vector store for this agent
        try:
            create_vector_store(splits, persist_dir, args.batch_size, verbose=args.verbose)
            print(f"Successfully processed agent '{agent_name}' into '{persist_dir}'")
        except Exception as e:
            print(f"Error creating vector store for agent {agent_name}: {e}")
            # Optionally continue to the next agent or stop
            # continue 
            raise # Stop on first error for now
            
    print("\nAll agent database processing finished.")

if __name__ == "__main__":
    main() 