import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load API keys once when the module is loaded
load_dotenv()

def get_openai_api_key():
    """Gets the OpenAI API key, ensuring it's loaded."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY_HERE_REPLACE_THIS":
        raise ValueError("OpenAI API key not found or is placeholder in environment/.env")
    return api_key

def create_specialist_rag_chain(db_path: str, llm_model: str = "gpt-4o", k_value: int = 5):
    """
    Creates a RAG chain specifically for a given vector database path.

    Args:
        db_path: Path to the domain-specific FAISS database directory.
        llm_model: The OpenAI model to use for generation.
        k_value: The number of documents to retrieve.

    Returns:
        A LangChain runnable chain.
    """
    # Ensure API key is available (consider better ways in production)
    get_openai_api_key()

    # Basic check if DB path exists
    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        print(f"Warning: Database path {db_path} or index.faiss not found.")
        # Return a dummy chain or raise error?
        # For now, let it fail during FAISS.load_local if path is truly invalid
        pass 

    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            folder_path=db_path,
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
    except Exception as e:
        print(f"Error loading vector store or creating retriever for {db_path}: {e}")
        # Return a chain that just explains the error
        return RunnableLambda(lambda x: f"Error accessing knowledge base for this topic ({os.path.basename(db_path)}): {e}")

    llm = ChatOpenAI(model=llm_model, temperature=0)

    # Generic prompt suitable for most specialist agents
    template = """You are an expert assistant for the specific topic derived from the user's question.
Answer the following question based *only* on the provided context snippets retrieved from the specialized knowledge base for this topic.
Do not use any prior knowledge outside of the provided context.
If the context does not contain the answer, state that the information is not available in this specific knowledge base.
Include relevant quotes or details from the context where appropriate.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs: List[Document]) -> str:
        """Helper function to format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Define the RAG pipeline
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("Testing agent_tools...")
    
    # Test with one of the created databases, e.g., db_general
    test_db_path = "./db_general"
    test_question = "What is the lien date?" 

    if not os.path.exists(os.path.join(test_db_path, "index.faiss")):
         print(f"Test database {test_db_path} not found. Cannot run test.")
    else:
        print(f"Creating RAG chain for: {test_db_path}")
        try:
            chain = create_specialist_rag_chain(test_db_path)
            print(f"Invoking chain with question: '{test_question}'")
            
            # Check if the chain is callable (not an error explanation)
            if isinstance(chain, RunnableLambda):
                 # Handle the case where the chain creation failed and returned an error message lambda
                 error_message = chain.invoke(test_question)
                 print(f"Chain creation failed: {error_message}")
            else:
                 answer = chain.invoke(test_question)
                 print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"Error during test: {e}") 