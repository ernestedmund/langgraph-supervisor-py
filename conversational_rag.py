from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, List, Optional
import os

# Set up your API key - you'll need to add yours here
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize models
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

# 1. LOADING DOCUMENTS
# For this example, we'll use mock documents about LangGraph and RAG
docs = [
    Document(
        page_content="LangGraph is a library for building stateful, multi-agent applications with LLMs. "
        "It provides a framework for creating complex agent workflows. LangGraph is built on top of LangChain "
        "and extends its capabilities with stateful execution and complex control flows.",
        metadata={"source": "langgraph_docs", "topic": "langgraph"}
    ),
    Document(
        page_content="Agents are systems that use language models to interact with other tools and APIs. "
        "They can solve complex tasks by breaking them down into smaller steps. LLM-based agents typically "
        "consist of a language model, a prompt, and tools that the agent can use.",
        metadata={"source": "langgraph_docs", "topic": "agents"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) enhances LLM outputs by incorporating "
        "relevant information retrieved from a knowledge base into the generation process. "
        "RAG systems typically involve an indexing phase (where documents are processed and stored) "
        "and a retrieval phase (where relevant documents are fetched based on a query).",
        metadata={"source": "langgraph_docs", "topic": "rag"}
    ),
    Document(
        page_content="Vector databases store embeddings of text chunks and allow for semantic similarity search. "
        "When a query is made, the query is embedded using the same model and the most similar chunks are retrieved. "
        "Popular vector databases include Chroma, FAISS, and Pinecone.",
        metadata={"source": "langgraph_docs", "topic": "vector_db"}
    ),
]

# 2. SPLITTING DOCUMENTS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. STORING DOCUMENTS
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(splits)

# 4. Define RAG prompt with conversation history
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that generates accurate, helpful responses based on the conversation history and retrieved context."),
    ("system", "Here's some relevant context to help answer the question:\n\n{context}"),
    # Include the conversation history
    ("placeholder", "{chat_history}"),
    # The latest user question
    ("user", "{question}")
])

# 5. Define state for the conversational RAG application
class State(TypedDict):
    question: str
    chat_history: List
    context: Optional[List[Document]]
    answer: Optional[str]

# 6. Define the component functions

def retrieve(state: State):
    """Retrieve relevant documents based on the question and chat history"""
    # Use the current question for retrieval
    retrieved_docs = vectorstore.similarity_search(
        state["question"],
        k=2  # Retrieve top 2 most relevant chunks
    )
    return {"context": retrieved_docs}

def generate(state: State):
    """Generate an answer based on the retrieved documents and chat history"""
    # Combine the content of all retrieved documents
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Format chat history for the prompt
    formatted_chat_history = []
    for message in state["chat_history"]:
        if isinstance(message, HumanMessage):
            formatted_chat_history.append(("user", message.content))
        elif isinstance(message, AIMessage):
            formatted_chat_history.append(("assistant", message.content))
    
    # Format the prompt with the question, context, and chat history
    messages = rag_prompt.invoke({
        "question": state["question"], 
        "context": docs_content,
        "chat_history": formatted_chat_history
    })
    
    # Generate a response using the LLM
    response = llm.invoke(messages)
    return {"answer": response.content}

def update_chat_history(state: State):
    """Update the chat history with the new question and answer"""
    # Add the new question and answer to the chat history
    return {
        "chat_history": state["chat_history"] + [
            HumanMessage(content=state["question"]),
            AIMessage(content=state["answer"])
        ]
    }

# 7. Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("update_chat_history", update_chat_history)

# Add edges
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", "update_chat_history")
graph_builder.add_edge("update_chat_history", END)

# Compile the graph
graph = graph_builder.compile()

# Function to run a conversation with the RAG system
def chat_with_rag():
    """Interactive chat function with the RAG system"""
    print("Conversational RAG System")
    print("------------------------")
    print("Type 'exit' to end the conversation\n")
    
    # Initialize the state with empty chat history
    state = {
        "chat_history": [],
        "question": "",
        "context": None,
        "answer": None
    }
    
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        # Update the question in the state
        state["question"] = user_input
        
        # Run the graph with the current state
        state = graph.invoke(state)
        
        # Display the answer
        print(f"Assistant: {state['answer']}\n")

if __name__ == "__main__":
    print("Conversational RAG Implementation")
    print("-------------------------------")
    print("Before running, make sure to add your OpenAI API key to the script.")
    
    if os.environ.get("OPENAI_API_KEY"):
        chat_with_rag()
    else:
        print("\nPlease add your OpenAI API key to the script to run the examples.") 