from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
import os

# Set up your API key - you'll need to add yours here
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create a model
model = ChatOpenAI(model="gpt-4o")

# Define tools for our agents
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def web_search(query: str) -> str:
    """Mock search function that returns predefined data."""
    return (
        "Here's some information about Python libraries:\n"
        "1. LangGraph: A library for building stateful, multi-agent applications with LLMs\n"
        "2. LangChain: A framework for developing applications powered by language models\n"
        "3. LangGraph Supervisor: A library for creating hierarchical multi-agent systems\n"
    )

# Create specialized agents
math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert who excels at calculations."
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="search_expert",
    prompt="You are a search expert who finds information."
)

# Create supervisor workflow
workflow = create_supervisor(
    [math_agent, research_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a math expert and a search expert. "
        "For calculations, use the math_expert. "
        "For information lookup, use the search_expert."
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
    print("LangGraph Supervisor Example")
    print("----------------------------")
    print("Before running, make sure to add your OpenAI API key to the script.")
    
    # Demo queries
    if os.environ.get("OPENAI_API_KEY"):
        run_query("What is 24 × 15?")
        run_query("Tell me about LangGraph libraries")
    else:
        print("\nPlease add your OpenAI API key to the script to run the examples.") 