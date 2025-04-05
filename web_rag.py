from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing import TypedDict, List
import bs4
import os

# Set up your API key - you'll need to add yours here
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize models
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

# 1. LOADING DOCUMENTS FROM WEB
# Use WebBaseLoader to load content from a blog post about LLM agents
loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

print("Loading documents from web...")
docs = loader.load()
print(f"Loaded {len(docs)} document(s)")

# 2. SPLITTING DOCUMENTS
# Split the documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Split into {len(splits)} chunks")

# 3. STORING DOCUMENTS
# Create a vector store and add the document chunks
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(splits)
print("Added documents to vector store")

# 4. Define RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that generates accurate, helpful responses based on the retrieved context."),
    ("user", "{question}"),
    ("system", "Here's some relevant context to help answer the question:\n\n{context}")
])

# 5. Define state for the RAG application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 6. Define the component functions

def retrieve(state: State):
    """Retrieve relevant documents based on the question"""
    # Perform a similarity search in the vector store
    retrieved_docs = vectorstore.similarity_search(
        state["question"],
        k=3  # Retrieve top 3 most relevant chunks
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

# 7. Build the graph - simpler than before, just retrieve and generate
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Function to run the RAG pipeline
def run_rag_query(question):
    """Run a query through the RAG pipeline"""
    print(f"\n--- Question: {question} ---")
    
    # Initialize the state with the question
    initial_state = {"question": question}
    
    # Process the query and get the result
    result = graph.invoke(initial_state)
    
    # Print the answer
    print(f"Answer: {result['answer']}")
    
    # Print the source chunks used (optional)
    print("\nSource chunks used:")
    for i, doc in enumerate(result['context']):
        print(f"Chunk {i+1} (excerpt): {doc.page_content[:100]}...")

if __name__ == "__main__":
    print("Web-based RAG Implementation")
    print("---------------------------")
    print("Before running, make sure to add your OpenAI API key to the script.")
    
    if os.environ.get("OPENAI_API_KEY"):
        # Test questions about the LLM agents blog post
        run_rag_query("What are the main components of an LLM-based agent?")
        run_rag_query("What are some challenges with LLM-based agents?")
        run_rag_query("Explain ReAct in the context of LLM agents")
    else:
        print("\nPlease add your OpenAI API key to the script to run the examples.") 