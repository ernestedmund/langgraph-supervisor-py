import os
import json
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Literal
import operator

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition

# Import the RAG chain creation function
from agent_tools import create_specialist_rag_chain, get_openai_api_key

# --- Agent State --- 
class AgentState(TypedDict):
    messages: Annotated[Sequence[Dict[str, Any]], operator.add]
    next_node: str # Tracks the next specialist agent to call

# --- Nodes --- 

# 1. Supervisor Node
# This node decides which specialist agent should handle the query next.

# Define the routing schema - Updated for consolidated agents
class RouteQuery(BaseModel):
    """Route the user's query to the most relevant consolidated specialist agent."""
    destination: Literal[
        "RealEstateAgent",
        "OwnershipTransferAgent",
        "BusinessPersonalAgent",
        "ExemptionsAppealsAgent",
        "FINISH" # If no specific agent is relevant or conversation should end
    ] = Field(
        ...,
        description="Given the user query, choose the *single* most relevant specialist agent (RealEstate, OwnershipTransfer, BusinessPersonal, ExemptionsAppeals) to handle it. If the query is unclear, a simple follow-up like 'thank you', or doesn't fit a specific category, route to FINISH.",
    )

# List of available consolidated database directories - Updated
# IMPORTANT: Ensure these names match the agent_sources keys and db_prefix used during import
consolidated_dbs = {
    "RealEstateAgent": "./db_RealEstateAgent",
    "OwnershipTransferAgent": "./db_OwnershipTransferAgent",
    "BusinessPersonalAgent": "./db_BusinessPersonalAgent",
    "ExemptionsAppealsAgent": "./db_ExemptionsAppealsAgent",
}

def create_supervisor_node():
    """Creates the supervisor node runnable."""
    get_openai_api_key()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create the routing prompt - Updated for consolidated agents
    system_prompt = (
        "You are a supervisor routing user queries to the correct specialist agent. "
        "Based on the user's query, determine which of the following consolidated agents is best suited to answer it. "
        "Do not answer the question yourself, only route it."
        "Available agents:\n"
        f"- RealEstateAgent: Handles questions about general real property (residential, commercial, manufactured housing), assessments, value changes (Prop 8), new construction, supplemental assessments, calamities, mapping/lot lines, restricted properties (CLCA), and general tax definitions/processes.\n"
        f"- OwnershipTransferAgent: Handles questions specifically about property transfers, changes in ownership, exclusions (parent-child/Prop 58/193, spousal/Prop 19), trusts, and legal entity transfers.\n"
        f"- BusinessPersonalAgent: Handles questions about business personal property assessments, 571-L forms, boats, and aircraft.\n"
        f"- ExemptionsAppealsAgent: Handles questions about property tax exemptions (homeowner, veteran, non-profit, religious, welfare, etc.) AND the assessment appeals process.\n\n"
        "Route to FINISH if the query is unclear, a simple follow-up like 'thank you', or doesn't seem to fit any category clearly."
        "\nUser Query:\n{query}"
    )
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    routing_chain = prompt | llm.with_structured_output(RouteQuery)
    
    # The node function
    def supervisor(state: AgentState):
        print("---SUPERVISOR--- ")
        last_message_content = state["messages"][-1]["content"]
        print(f"Routing query: '{last_message_content}'")
        
        # Invoke the combined chain
        route = routing_chain.invoke({"query": last_message_content})
        
        print(f"Decision: Route to {route.destination}")
        return {"next_node": route.destination}
        
    return supervisor

# 2. Specialist Agent Nodes
# We create these dynamically based on the found databases
def create_specialist_node(agent_name: str, db_path: str):
    """Creates a node that runs the RAG chain for a specific agent."""
    # Create the chain *once* when the node function is defined
    rag_chain = create_specialist_rag_chain(db_path) 
    
    def agent_node(state: AgentState):
        print(f"---{agent_name}--- ")
        query = state["messages"][-1]["content"]
        print(f"Processing query: '{query}'")
        
        # Simplified invocation and error handling
        try:
            # Directly invoke the chain returned by the factory function
            answer = rag_chain.invoke(query)
            print(f"Generated answer.")
            
            # Check if the answer indicates an internal error from chain creation
            if isinstance(answer, str) and answer.startswith("Error accessing knowledge base"):
                print(f"Warning: RAG chain creation failed for {agent_name}: {answer}")
                # Pass the specific error message along
                
        except Exception as e:
            # Catch errors during the actual invocation
            print(f"Error invoking RAG chain for {agent_name}: {e}")
            answer = f"An error occurred while processing the request with {agent_name}. Details: {e}"
            
        # Return response
        return {"messages": [{"role": agent_name, "content": answer}]}
        
    return agent_node

# --- Graph Setup --- 

# Initialize the graph state
workflow = StateGraph(AgentState)

# Create and add the supervisor node
supervisor_node = create_supervisor_node()
workflow.add_node("Supervisor", supervisor_node)

# Create and add nodes for each CONSOLIDATED specialist agent found
specialist_nodes = {}
# Use the consolidated_dbs dictionary now
for agent_name, db_path in consolidated_dbs.items(): 
    # Check based on the consolidated DB path format
    if os.path.exists(os.path.join(db_path, "index.faiss")):
        print(f"Creating node for {agent_name} using DB {db_path}")
        node_func = create_specialist_node(agent_name, db_path)
        workflow.add_node(agent_name, node_func)
        specialist_nodes[agent_name] = node_func 
    else:
        print(f"Skipping node creation for {agent_name}: Consolidated DB not found at {db_path}")

if not specialist_nodes:
    raise ValueError("No consolidated specialist agent databases found! Cannot build graph.")

# Define conditional edges (routing logic is the same, uses new agent names)
def route_to_specialist(state: AgentState):
    destination = state.get("next_node")
    if destination == "FINISH" or destination not in specialist_nodes:
        print("Routing to END")
        return END
    else:
        print(f"Routing to {destination}")
        return destination

workflow.add_conditional_edges(
    "Supervisor",
    route_to_specialist,
    list(specialist_nodes.keys()) + [END] # Uses keys from consolidated_dbs
)

# Define edges: route specialists to END (prevents loops for now)
for agent_name in specialist_nodes.keys():
     workflow.add_edge(agent_name, END) 

workflow.set_entry_point("Supervisor")

# Compile the graph
app = workflow.compile()
print("\nGraph compiled successfully with CONSOLIDATED agents!")

# --- Running the Graph --- 

if __name__ == "__main__":
    print("\nEnter your query to start the agent conversation.")
    print("Type 'exit', 'quit', or 'q' to end.")

    while True:
        user_input = input("\nUser Query: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        
        # Prepare initial state
        initial_state = {"messages": [{"role": "user", "content": user_input}]}
        
        print("---Invoking Graph--- ")
        final_state = app.invoke(initial_state)
        
        # Print the final response (likely the last message from a specialist agent)
        if final_state and "messages" in final_state and final_state["messages"]:
            last_agent_message = final_state["messages"][-1]
            print(f"\nFinal Answer ({last_agent_message.get('role', 'Agent')}):")
            print(last_agent_message.get("content", "No content found."))
        else:
            print("\nNo final answer generated by the graph.")

    print("\nConversation ended.") 