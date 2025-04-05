from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing import TypedDict, List, Annotated, Literal
import os

# Set up your API key - you'll need to add yours here
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize models
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

# 1. LOADING DOCUMENTS
# Load a sample document - in a real application you would load from files, URLs, etc.
# For this example, we'll create a simple mock document about LangGraph
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
]

# 2. SPLITTING DOCUMENTS
# Split the documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Update metadata (for demonstration purposes)
for i, document in enumerate(splits):
    if i < len(splits) // 3:
        document.metadata["section"] = "beginning"
    elif i < 2 * len(splits) // 3:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

# 3. STORING DOCUMENTS
# Create a vector store and add the document chunks
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(splits)

# 4. Define query schema
class Search(TypedDict):
    """Search query."""
    query: Annotated[str, "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        "Section to query.",
    ]

# 5. Define RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that generates accurate, helpful responses based on the retrieved context."),
    ("user", "{question}"),
    ("system", "Here's some relevant context to help answer the question:\n\n{context}")
])

# 6. Define state for the RAG application
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

# 7. Define the component functions

def analyze_query(state: State):
    """Analyze the user's question to determine the best search parameters"""
    # Use structured output to create a search query
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

def retrieve(state: State):
    """Retrieve relevant documents based on the query"""
    query = state["query"]
    # Use the filter to find documents from the right section
    retrieved_docs = vectorstore.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}

def generate(state: State):
    """Generate an answer based on the retrieved documents"""
    # Combine the content of all retrieved documents
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # Format the prompt with the question and context
    messages = rag_prompt.invoke({"question": state["question"], "context": docs_content})
    # Generate a response using the LLM
    response = llm.invoke(messages)
    return {"answer": response.content}

# 8. Build the graph
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# Function to run the RAG pipeline
def run_rag_query(question):
    """Run a query through the RAG pipeline"""
    print(f"\n--- Question: {question} ---")
    
    # Initialize the state with the question
    initial_state = {"question": question}
    
    # Process the query and get the result
    for step in graph.stream(initial_state, stream_mode="updates"):
        # Print each step in the process for clarity
        step_name = list(step.keys())[0] if step else "No step"
        print(f"Step: {step_name}")
        if step_name == "generate":
            print(f"Answer: {step['generate']['answer']}")

if __name__ == "__main__":
    print("Basic RAG Implementation")
    print("----------------------")
    print("Before running, make sure to add your OpenAI API key to the script.")
    
    if os.environ.get("OPENAI_API_KEY"):
        # Test questions
        run_rag_query("What does the beginning section say about LangGraph?")
        run_rag_query("Tell me about agents from the middle section.")
        run_rag_query("What is RAG according to the end section?")
    else:
        print("\nPlease add your OpenAI API key to the script to run the examples.") 