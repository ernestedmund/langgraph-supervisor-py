import os
import json
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Literal
import operator

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    """Creates the supervisor node runnable, considering chat history."""
    get_openai_api_key()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create the routing prompt - Updated to accept history
    system_prompt = (
        "You are a supervisor routing user queries and conversation turns to the correct specialist agent. "
        "Based on the CURRENT user query AND the PRECEDING conversation history, determine which agent should handle the next step. "
        "Do not answer the question yourself, only route it."
        "Available agents:\n"
        f"- RealEstateAgent: Handles questions about general real property (residential, commercial, manufactured housing), assessments, value changes (Prop 8), new construction, supplemental assessments, calamities, mapping/lot lines, restricted properties (CLCA), and general tax definitions/processes.\n"
        f"- OwnershipTransferAgent: Handles questions specifically about property transfers, changes in ownership, exclusions (parent-child/Prop 58/193, spousal/Prop 19), trusts, and legal entity transfers.\n"
        f"- BusinessPersonalAgent: Handles questions about business personal property assessments, 571-L forms, boats, and aircraft.\n"
        f"- ExemptionsAppealsAgent: Handles questions about property tax exemptions (homeowner, veteran, non-profit, religious, welfare, etc.) AND the assessment appeals process.\n\n"
        "Routing Rules:"
        "- If the CURRENT query looks like an answer from another agent, route to FINISH. Do NOT route an agent's answer back to another agent."
        "- If the user asks a new question fitting a specific category, route to that agent."
        "- If the user asks a follow-up question, consider the history and route to the agent relevant to the *original topic* or the agent that last spoke, if appropriate."
        "- Route to FINISH if the query is unclear, a simple closing like 'thank you' or 'okay', or if the conversation seems complete."
        "\nConversation History (newest messages last):\n{history}"
        "\nCURRENT Query to Route:\n{query}" # Renamed for clarity
    )
    
    # Use ChatPromptTemplate for easier message handling
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        # Placeholder for history - LangGraph handles this, but we structure the prompt this way
        # We will format the history before passing it to the chain.
    ])
    
    routing_chain = prompt | llm.with_structured_output(RouteQuery)
    
    def format_history_for_prompt(messages: Sequence[Dict[str, Any]]) -> str:
        """Formats the message history for the supervisor prompt."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Simple formatting, adjust as needed
            formatted.append(f"{role.capitalize()}: {content}") 
        return "\n".join(formatted)

    # The node function
    def supervisor(state: AgentState):
        print("---SUPERVISOR--- ")
        all_messages = state["messages"]
        # The latest message is the query we need to route
        current_query = all_messages[-1]["content"]
        # The history is everything *before* the latest message
        history_messages = all_messages[:-1]
        
        formatted_history = format_history_for_prompt(history_messages)
        
        print(f"Routing query: '{current_query}'")
        print(f"With history: \n{formatted_history}")
        
        # Invoke the chain with history and the current query
        route = routing_chain.invoke({
            "history": formatted_history,
            "query": current_query 
        })
        
        print(f"Decision: Route to {route.destination}")
        return {"next_node": route.destination}
        
    return supervisor

# 2. Specialist Agent Nodes
def create_specialist_node(agent_name: str, db_path: str):
    """Creates a node that runs the RAG chain for a specific agent."""
    rag_chain = create_specialist_rag_chain(db_path)
    
    def agent_node(state: AgentState):
        print(f"---{agent_name}--- ")
        # Get the full message history
        messages = state["messages"]
        # The user query is the last message
        query = messages[-1]["content"]
        # The history is everything before the last message
        chat_history = messages[:-1]
        
        print(f"Processing query: '{query}'")
        if chat_history:
             print(f"With history: {len(chat_history)} previous messages")
        
        # Prepare input dictionary for the RAG chain
        rag_input = {"question": query, "chat_history": chat_history}
        
        try:
            # Invoke the chain with the input dictionary
            answer = rag_chain.invoke(rag_input)
            print(f"Generated answer.")
            
            if isinstance(answer, str) and answer.startswith("Error accessing knowledge base"):
                print(f"Warning: RAG chain creation failed for {agent_name}: {answer}")
                
        except Exception as e:
            print(f"Error invoking RAG chain for {agent_name}: {e}")
            # Format exception details into the answer
            import traceback
            tb_str = traceback.format_exc()
            answer = f"An error occurred while processing the request with {agent_name}. Details: {e}\n{tb_str}"
            
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

# Define conditional edges: after supervisor, route to the chosen specialist or end
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
    list(specialist_nodes.keys()) + [END]
)

# Define edges: after any specialist agent runs, route back to supervisor
# This allows for conversational follow-up.
for agent_name in specialist_nodes.keys():
     workflow.add_edge(agent_name, "Supervisor") # Changed back from END

workflow.set_entry_point("Supervisor")

# Compile the graph
app = workflow.compile()
print("\nGraph compiled successfully with CONSOLIDATED agents!")

# --- Running the Graph --- 

if __name__ == "__main__":
    print("\nEnter your query to start the agent conversation.")
    print("Type 'exit', 'quit', or 'q' to end.")

    # Maintain the conversation history messages here
    conversation_messages = []

    while True:
        user_input = input("\nUser Query: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        
        # Add the new user message to the conversation history
        conversation_messages.append({"role": "user", "content": user_input})
        
        # Prepare state using the CURRENT conversation history
        current_state = {"messages": conversation_messages}
        
        print("---Invoking Graph--- ")
        # Invoke the graph with the current state
        final_state = app.invoke(current_state)
        
        # Update conversation history with the messages generated by the graph run
        # Note: LangGraph state includes *all* messages, including the input user message
        # We need to store the final state's messages for the next turn.
        conversation_messages = final_state.get("messages", []) 
        
        # Print the final response (the last message added by the graph)
        if conversation_messages:
            last_agent_message = conversation_messages[-1]
            # Avoid printing if the last message was just the user input (e.g., if graph routed straight to END)
            if last_agent_message.get("role") != "user":
                 print(f"\nFinal Answer ({last_agent_message.get('role', 'Agent')}):")
                 print(last_agent_message.get("content", "No content found."))
            # Handle cases where the graph might end without an agent response? Maybe just pass.
        else:
            print("\nGraph execution finished, but no messages found in the final state.")

    print("\nConversation ended.") 