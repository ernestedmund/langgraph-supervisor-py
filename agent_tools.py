import os
from typing import List, Dict, Any, Sequence
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load API keys once when the module is loaded
load_dotenv()

def get_openai_api_key():
    """Gets the OpenAI API key, ensuring it's loaded."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY_HERE_REPLACE_THIS":
        raise ValueError("OpenAI API key not found or is placeholder in environment/.env")
    return api_key

def format_chat_history(chat_history: Sequence[Dict[str, Any]]) -> List:
    """Formats the dictionary history into a list of BaseMessage objects."""
    buffer = []
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            buffer.append(HumanMessage(content=content))
        elif role and role != "user": # Treat supervisor or other agents as AIMessage
            buffer.append(AIMessage(content=content))
        # else: skip if role/content missing or unknown?
    return buffer

def create_condensed_question_chain(llm_model: str = "gpt-3.5-turbo"): # Use cheaper model for this
    """Creates a chain to condense chat history and follow-up question into a standalone question."""
    condense_q_system_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. Include necessary context from the chat history."""
    condense_q_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    condense_q_chain = condense_q_prompt | ChatOpenAI(temperature=0, model=llm_model) | StrOutputParser()
    return condense_q_chain

def create_specialist_rag_chain(db_path: str, llm_model: str = "gpt-4o", k_value: int = 5):
    """
    Creates a RAG chain specifically for a given vector database path, 
    considering chat history.
    
    Accepts a dictionary with "question" and "chat_history" keys.
    """
    get_openai_api_key()

    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        print(f"Warning: Database path {db_path} or index.faiss not found.")
        return RunnableLambda(lambda x: f"Error accessing knowledge base for this topic ({os.path.basename(db_path)}): DB not found")

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
        return RunnableLambda(lambda x: f"Error accessing knowledge base for this topic ({os.path.basename(db_path)}): {e}")

    llm = ChatOpenAI(model=llm_model, temperature=0)
    condense_question_chain = create_condensed_question_chain()

    # Answer prompt now includes placeholder for chat history
    template = """You are an expert assistant for the specific topic derived from the user's question.
Answer the following question based *only* on the provided context snippets retrieved from the specialized knowledge base for this topic.
Do not use any prior knowledge outside of the provided context.
If the context does not contain the answer, state that the information is not available in this specific knowledge base.
Include relevant quotes or details from the context where appropriate.

Context:
{context}

Question: {question} 

Answer:"""
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history") # Include history here for final answer generation
    ])

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
        
    def route_to_retriever(inputs: Dict):
        """Decides whether to run condense_question_chain or use original question."""
        if inputs.get("chat_history"): # If history exists, condense
            return condense_question_chain
        else: # Otherwise, just pass the original question through
            return RunnableLambda(lambda x: x["question"])

    # Chain to handle potentially condensing the question based on history
    # It takes the full input dict and outputs the question string for the retriever
    condensed_question_for_retrieval = RunnableLambda(route_to_retriever) | { # Pass input dict to route_to_retriever
         # Output of route_to_retriever (either condense chain or lambda) is invoked here
         # Input to the condense chain: {"question": ..., "chat_history": ...}
         # Input to the lambda: {"question": ...}
         # Output needed by retriever: just the question string
         "question_string": lambda x: x # Assuming the output is now the string
     }
     
    # Modified RAG Chain to handle history
    rag_chain = (
        RunnablePassthrough.assign( # Keep original inputs accessible
            # Condense question using chat history (handles empty history correctly)
            condensed_question=lambda x: condense_question_chain.invoke({
                "question": x["question"],
                "chat_history": format_chat_history(x.get("chat_history", [])) # Pass empty list if no history
            })
        )
        # Correctly pipe the condensed question string to the retriever chain
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: x["condensed_question"]) | retriever | format_docs
        )
        | {
            # Prepare final prompt inputs
            "context": lambda x: x["context"],
            "question": lambda x: x["condensed_question"], 
            "chat_history": lambda x: format_chat_history(x.get("chat_history", []))
          }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Example usage update
if __name__ == '__main__':
    print("Testing agent_tools with history...")
    test_db_path = "./db_general"
    
    if not os.path.exists(os.path.join(test_db_path, "index.faiss")):
         print(f"Test database {test_db_path} not found. Cannot run test.")
    else:
        print(f"Creating RAG chain for: {test_db_path}")
        try:
            chain = create_specialist_rag_chain(test_db_path)
            print("Invoking chain with initial question...")
            
            initial_question = "What is the standard assessment date?"
            # Example input format with history (empty for first turn)
            input_data_1 = {"question": initial_question, "chat_history": []}
            answer1 = chain.invoke(input_data_1)
            print(f"\nQ1: {initial_question}")
            print(f"A1: {answer1}")
            
            print("\nInvoking chain with follow-up question...")
            follow_up_question = "Why that specific date?"
            # Example history (manually constructed for test)
            history = [
                {"role": "user", "content": initial_question},
                {"role": "agent", "content": answer1} # Assuming agent role for simplicity
            ]
            input_data_2 = {"question": follow_up_question, "chat_history": history}
            answer2 = chain.invoke(input_data_2)
            print(f"\nQ2 (Follow-up): {follow_up_question}")
            print(f"A2: {answer2}")

        except Exception as e:
            print(f"Error during test: {e}") 