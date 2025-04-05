from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
import os

# Set up your API key - you'll need to add yours here
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create a model
model = ChatOpenAI(model="gpt-4o")

# Define tools
def web_search(query: str) -> str:
    """Mock search function that returns predefined data."""
    return (
        "Information about the weather:\n"
        "- New York: 72°F, Sunny\n"
        "- London: 65°F, Cloudy with light rain\n"
        "- Tokyo: 80°F, Clear skies\n"
        "- Sydney: 68°F, Partly cloudy\n"
    )

# Create the first agent using the Graph API (React agent pattern)
research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a search expert who finds information."
)

# Create a second agent using the Functional API (more flexible)
@task
def generate_joke(messages):
    """Generate a joke based on the input"""
    from langchain_core.messages import SystemMessage
    
    system_message = SystemMessage(content="Write a short, clever joke related to the user's request.")
    result = model.invoke([system_message] + messages)
    return result

@entrypoint()
def joke_agent(state):
    """Joke agent that generates jokes based on the input"""
    joke = generate_joke(state['messages']).result()
    messages = add_messages(state["messages"], [joke])
    return {"messages": messages}

# Set the name of the functional agent
joke_agent.name = "joke_agent"

# Create supervisor workflow combining both agent types
workflow = create_supervisor(
    [research_agent, joke_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a joke expert. "
        "For factual information, use the research_expert. "
        "For humor and jokes, use the joke_agent."
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
    print("LangGraph Supervisor with Functional API Example")
    print("-----------------------------------------------")
    print("Before running, make sure to add your OpenAI API key to the script.")
    
    # Demo queries
    if os.environ.get("OPENAI_API_KEY"):
        run_query("Tell me a joke about programmers")
        run_query("What's the weather like in major cities?")
    else:
        print("\nPlease add your OpenAI API key to the script to run the examples.") 