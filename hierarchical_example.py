from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
import os

# Set up your API key - you'll need to add yours here
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create a model
model = ChatOpenAI(model="gpt-4o")

# Define tools for different agents
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

def write_poem(topic: str) -> str:
    """Simulate writing a poem by returning a pre-written one."""
    return f"Here's a poem about {topic}:\n\nIn the realm of code and light,\nWhere algorithms take their flight,\nPython's grace, a gentle might,\nGuiding us through day and night."

def summarize(text: str) -> str:
    """Simulate text summarization."""
    return f"Summary: {text.split('.')[0]}."

# --------- Level 1: Specialized Agents ---------

# Math expert
math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert who excels at calculations."
)

# Research expert
research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a research expert who finds information through web searches."
)

# Creative writer
creative_agent = create_react_agent(
    model=model,
    tools=[write_poem],
    name="creative_expert",
    prompt="You are a creative expert who can write beautiful poems and stories."
)

# Editor
editor_agent = create_react_agent(
    model=model,
    tools=[summarize],
    name="editor_expert",
    prompt="You are an editor who excels at summarizing and refining content."
)

# --------- Level 2: Team Supervisors ---------

# Technical team supervisor (manages math and research)
tech_team = create_supervisor(
    [math_agent, research_agent],
    model=model,
    supervisor_name="tech_supervisor",
    prompt=(
        "You are a technical team supervisor managing a math expert and a research expert. "
        "For calculations, use the math_expert. "
        "For information lookup, use the research_expert."
    )
).compile(name="tech_team")

# Creative team supervisor (manages creative writer and editor)
creative_team = create_supervisor(
    [creative_agent, editor_agent],
    model=model,
    supervisor_name="creative_supervisor",
    prompt=(
        "You are a creative team supervisor managing a creative writer and an editor. "
        "For writing poems or stories, use the creative_expert. "
        "For editing and summarizing content, use the editor_expert."
    )
).compile(name="creative_team")

# --------- Level 3: Top-level Supervisor ---------

# CEO (manages both teams)
top_level_supervisor = create_supervisor(
    [tech_team, creative_team],
    model=model,
    supervisor_name="ceo",
    prompt=(
        "You are the CEO managing a technical team and a creative team. "
        "For technical questions, calculations, or research, use the tech_team. "
        "For creative content, writing, or editing, use the creative_team. "
        "Analyze the user's request and delegate to the appropriate team."
    )
).compile(name="ceo")

def run_query(query):
    """Helper function to run a query and print the results"""
    print(f"\n--- Query: {query} ---")
    result = top_level_supervisor.invoke({
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
    print("Hierarchical Multi-Agent System Example")
    print("--------------------------------------")
    print("Before running, make sure to add your OpenAI API key to the script.")
    
    # Demo queries
    if os.environ.get("OPENAI_API_KEY"):
        run_query("Can you calculate 25 × 16 for me?")
        run_query("Tell me about LangGraph")
        run_query("Write a poem about programming")
        run_query("First research LangGraph and then summarize the information")
    else:
        print("\nPlease add your OpenAI API key to the script to run the examples.") 