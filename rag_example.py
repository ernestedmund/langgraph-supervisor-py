from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
import os
from typing import List, Dict, Any

# Set up your API key - you'll need to add yours here
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create models
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

# For simplicity in this example, we'll create some mock documents
docs = [
    Document(
        page_content="LangGraph is a library for building stateful, multi-agent applications with LLMs. "
        "It provides a framework for creating complex agent workflows.",
        metadata={"source": "langgraph_docs", "section": "overview"}
    ),
    Document(
        page_content="Agents are systems that use language models to interact with other tools and APIs. "
        "They can solve complex tasks by breaking them down into smaller steps.",
        metadata={"source": "langgraph_docs", "section": "agents"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) enhances LLM outputs by incorporating "
        "relevant information retrieved from a knowledge base into the generation process.",
        metadata={"source": "langgraph_docs", "section": "rag"}
    ),
    Document(
        page_content="Supervisors are special agents that coordinate the work of other agents. "
        "They decide which agent to delegate tasks to based on the current context.",
        metadata={"source": "langgraph_docs", "section": "supervisors"}
    ),
]

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Create a vector store and add the chunks
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)

# Create a retrieval function
def retrieve_documents(query: str, k: int = 3) -> str:
    """Retrieve relevant documents for a query."""
    documents = vector_store.similarity_search(query, k=k)
    if not documents:
        return "No relevant documents found."
    
    # Format documents for the agent
    formatted_docs = []
    for i, doc in enumerate(documents):
        formatted_docs.append(f"Document {i+1}:\n{doc.page_content}\nSource: {doc.metadata.get('source', 'unknown')}")
    
    return "\n\n".join(formatted_docs)

# Define RAG-specific tools for our agents

# Tool for the retriever agent
def get_context(query: str) -> str:
    """Retrieve relevant information from the knowledge base."""
    return retrieve_documents(query)

# Tool for the answer generator agent
def answer_with_context(question: str, context: str) -> str:
    """Generate an answer based on the question and provided context."""
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that generates accurate, helpful responses based on the retrieved context."),
        ("user", "{question}"),
        ("system", "Here's some relevant context to help answer the question:\n\n{context}")
    ])
    
    messages = rag_prompt.invoke({"question": question, "context": context})
    response = llm.invoke(messages)
    return response.content

# Create specialized agents

# 1. Retriever Agent: Responsible for finding relevant information
retriever_agent = create_react_agent(
    model=llm,
    tools=[get_context],
    name="retriever_agent",
    prompt=(
        "You are a retrieval expert. Your job is to find relevant information for the user's question. "
        "Always use the get_context tool to retrieve information. "
        "Be thorough and make sure you're retrieving information that's most relevant to the query."
    )
)

# 2. Answer Generator Agent: Responsible for creating responses based on context
answer_agent = create_react_agent(
    model=llm,
    tools=[answer_with_context],
    name="answer_agent",
    prompt=(
        "You are an answer generation expert. Your job is to generate accurate, helpful answers "
        "based on the context provided by the retriever. "
        "Always use the answer_with_context tool to generate your final answer. "
        "Be sure to use both the original question and the retrieved context when generating your response."
    )
)

# Create the supervisor workflow that coordinates both agents
workflow = create_supervisor(
    [retriever_agent, answer_agent],
    model=llm,
    prompt=(
        "You are a RAG system supervisor managing a retriever agent and an answer generator agent. "
        "Your job is to coordinate a two-step retrieval-augmented generation process:\n"
        "1. First, ALWAYS use the retriever_agent to find relevant context for the user's question.\n"
        "2. Then, ALWAYS use the answer_agent to generate a final answer based on the retrieved context.\n"
        "Follow this exact sequence for all user questions - first retrieval, then generation."
    )
)

# Compile the workflow
app = workflow.compile()

def run_query(query):
    """Helper function to run a query and print the results"""
    print(f"\n--- Query: {query} ---")
    result = app.invoke({
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    })
    
    # Print the results
    for m in result["messages"]:
        print(f"{m.role}: {m.content if hasattr(m, 'content') and m.content else '[No content - tool call]'}")

if __name__ == "__main__":
    print("RAG with LangGraph Supervisor Example")
    print("-------------------------------------")
    print("Before running, make sure to add your OpenAI API key to the script.")
    
    # Demo queries
    if os.environ.get("OPENAI_API_KEY"):
        run_query("What is LangGraph?")
        run_query("How do supervisors work in multi-agent systems?")
        run_query("Explain Retrieval-Augmented Generation")
    else:
        print("\nPlease add your OpenAI API key to the script to run the examples.") 