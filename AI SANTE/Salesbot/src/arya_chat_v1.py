import os
import re
import logging
from typing import List, Literal, TypedDict
from dotenv import load_dotenv
 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
_log = logging.getLogger(__name__)

# -------------------- Load Environment Variables --------------------
load_dotenv()
_log.info("Environment variables loaded.")

# -------------------- Configuration --------------------
# Company and Chatbot Details
COMPANY_NAME = "AI Sante"
# COMPANY_DESCRIPTION = "a cutting-edge technology company dedicated to revolutionizing the pharmaceutical and healthcare industries through innovative AI-powered solutions. We specialize in developing smart, intuitive software that enhances business processes, optimizes decision-making, and improves operational efficiency for pharma professionals and businesses worldwide."
COMPANY_DESCRIPTION = "AI Sante is an innovative AI technology company focused on transforming the pharmaceutical and healthcare industries. It builds intelligent, intuitive software solutions—like RxIntel AI (a Pharma CRM)—that enhance operational efficiency, support data-driven decision-making, and empower pharmaceutical professionals to improve productivity and drive sales growth. With deep industry expertise and a customer-centric approach, AI Sante delivers tailored, AI-powered tools designed to meet the real-world needs of pharma businesses worldwide."
CORE_USPS="HRBotX, RCPA-Ai, Smart Pharma CRM, Sci-Coach Ai, DCR, CME, Ai Voice Command and Ai ChatBot"
COMPANY_PRODUCTS="Smart Pharma CRM, RxintelAi, Prescription_bot, LMSBot, HRBotX, General Bot, ChemistBot, Voice Command, Automated Expense, DCR (Smart Daily Call Reporting)"
CHATBOT_NAME = "Arya"

# LLM Setup
LLM_MODEL_NAME = "gemini-2.5-flash" # Using latest flash model
LLM_TEMPERATURE = 0.0 # For more deterministic and factual responses

# ChromaDB Setup (Must match your 01_pdf_chroma_gen.py)
CHROMA_DB = "chroma_db"
COLLECTION_NAME = "data_collection"
CHROMA_DB_DIR = f"{CHROMA_DB}/{COLLECTION_NAME}"

# -------------------- Initialize Components --------------------

# LLM
try:
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    _log.info(f"LLM initialized: {LLM_MODEL_NAME}")
except Exception as e:
    _log.critical(f"Failed to initialize LLM. Ensure GOOGLE_API_KEY is set in .env. Error: {e}")
    exit(1)

# Embeddings (Must match what you used to create the ChromaDB)
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    cache_folder="../huggingface_model" # Ensure this path is correct relative to where you run the script
)
_log.info(f"Embeddings model loaded: {embedding_model_name}")

# ChromaDB Vector Store
try:
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) # Retrieve top 10 relevant documents
    _log.info(f"ChromaDB vector store loaded from {CHROMA_DB_DIR}")
except Exception as e:
    _log.critical(f"Failed to load ChromaDB. Ensure it's populated. Error: {e}")
    exit(1)

# -------------------- LangGraph State Definition --------------------
class ChatbotState(TypedDict):
    """
    Represents the state of our chatbot.
    """
    chat_history: List[BaseMessage] # Stores the ongoing conversation
    user_query: str                # The current input from the user
    intent: Literal["greeting", "small_talk", "general_query", "sales_query", "unknown"] # Classified intent
    retrieved_docs: List[str]      # Content of documents retrieved from ChromaDB
    response: str                  # The chatbot's generated response

# -------------------- LangGraph Nodes (Functions) --------------------

def classify_intent(state: ChatbotState) -> ChatbotState:
    """
    Classifies the user's intent based on their query and chat history.
    """
    _log.info("Node: classify_intent")
    user_query = state["user_query"]
    chat_history = state["chat_history"]

    # Prompt for intent classification
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an intent classifier for {COMPANY_NAME}'s sales chatbot, {CHATBOT_NAME}.
        Classify the user's query into one of the following categories:
        - 'greeting': User is saying hello, good morning, etc.
        - 'small_talk': User is making casual conversation, asking about weather, how you are, etc.
        - 'general_query': User is asking a non-sales related question, e.g., about general AI, technology, or anything not directly related to {COMPANY_NAME}.
        - 'sales_query': User is asking about {COMPANY_NAME}, its products (like RxIntel AI), features, pricing, benefits, or anything related to a potential business inquiry.
        - 'unknown': If the intent is unclear or doesn't fit other categories.

        Provide only the category name as your response (e.g., 'sales_query')."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_query}")
    ])

    chain = prompt | llm | StrOutputParser()
    
    # Pass a limited chat history for intent classification to focus on recent context
    response = chain.invoke({"user_query": user_query, "chat_history": chat_history[-10:]}) # Last 10 messages for context

    # Clean and validate the intent
    intent = response.strip().lower()
    valid_intents = ["greeting", "small_talk", "general_query", "sales_query", "unknown"]
    if intent not in valid_intents:
        intent = "unknown" # Default to unknown if LLM hallucinates
    
    _log.info(f"Classified intent: {intent}")
    return {"intent": intent, "user_query": user_query, "chat_history": chat_history}

def handle_greeting_smalltalk_general(state: ChatbotState) -> ChatbotState:
    """
    Generates a direct, human-like response for non-sales queries.
    """
    _log.info("Node: handle_greeting_smalltalk_general")
    user_query = state["user_query"]
    chat_history = state["chat_history"]
    intent = state["intent"]

    # Persona and instructions for non-sales responses
    system_prompt = f"""You are {CHATBOT_NAME}, a friendly and helpful assistant for {COMPANY_NAME}.
    Your purpose is to engage in natural conversation. Do not provide sales information or mention specific products unless directly asked.
    Keep your responses brief, natural, and to the point.
    The user's intent has been classified as '{intent}'.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_query}")
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"user_query": user_query, "chat_history": chat_history})
    
    _log.info(f"Generated general response: {response[:50]}...")
    return {"response": response, "user_query": user_query, "chat_history": chat_history, "intent": intent}

def retrieve_sales_info(state: ChatbotState) -> ChatbotState:
    """
    Retrieves relevant documents from ChromaDB for sales queries.
    """
    _log.info("Node: retrieve_sales_info")
    user_query = state["user_query"]
    chat_history = state["chat_history"]

    # Optional: Query Rephrasing for better retrieval in conversational context
    # This helps if the user's query is highly contextual or relies on previous turns.
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful assistant. Given the following conversation and a follow-up question,
        rephrase the follow-up question to be a standalone question that can be used to retrieve information from a knowledge base.
        Do NOT answer the question, just rephrase it if necessary."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_query}")
    ])
    rephrase_chain = rephrase_prompt | llm | StrOutputParser()
    
    # Use the last few messages for rephrasing context
    rephrased_query = rephrase_chain.invoke({"user_query": user_query, "chat_history": chat_history[-10:]})
    _log.info(f"Original query: '{user_query}' | Rephrased query for retrieval: '{rephrased_query}'")

    docs = retriever.invoke(rephrased_query)
    retrieved_docs_content = [doc.page_content for doc in docs]
    
    _log.info(f"Retrieved {len(retrieved_docs_content)} documents.")
    return {"retrieved_docs": retrieved_docs_content, "user_query": user_query, "chat_history": chat_history, "intent": state["intent"]}

def generate_sales_response(state: ChatbotState) -> ChatbotState:
    """
    Generates a concise, WhatsApp-friendly, human-like sales response with product suggestions and CTA.
    """
    _log.info("Node: generate_sales_response")
    user_query = state["user_query"]
    chat_history = state["chat_history"]
    retrieved_docs = state["retrieved_docs"]

    # Combine retrieved documents into a single context string
    context = "\n\n".join(retrieved_docs)
    if not context:
        context = (
            "No specific information found in our knowledge base. "
            "I’ll try to help using general info about our product lineup."
        )

    # Sales Persona Prompt
    system_prompt = f"""
You are {CHATBOT_NAME}, a professional, friendly, and helpful sales consultant working for {COMPANY_NAME}.
{COMPANY_NAME} is: {COMPANY_DESCRIPTION}
Our core usps: {CORE_USPS}
Our available AI-powered products: {COMPANY_PRODUCTS}

Your goals:
- Help the user identify the best-fit product from the list above based on their query.
- Mention ONE primary recommended product clearly, based on context.
- Optionally suggest up to TWO other relevant tools briefly.
- Always end the message with a short, polite CTA to book a demo.
- Respond naturally, like a human—not like an assistant or AI.
- Keep responses short and clean — suitable for WhatsApp chat.
- Do not use symbols like *, **, bullet points, or long paragraphs.

Your answer must be:
- 2 to 3 short paragraphs.
- Focused, helpful, and sales-oriented.
- Clear and easy to read.

Here’s context from the knowledge base:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.strip()),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_query}")
    ])

    chain = prompt | llm | StrOutputParser()
    raw_response = chain.invoke({"user_query": user_query, "chat_history": chat_history})

    # Clean formatting (remove *, double newlines, etc.)
    cleaned_response = (
        raw_response.replace("*", "")
        .replace("**", "")
        .replace("•", "-")
        .replace("\n\n", "\n")
        .strip()
    )

    _log.info(f"Generated sales response: {cleaned_response[:50]}...")
    return {
        "response": cleaned_response,
        "user_query": user_query,
        "chat_history": chat_history,
        "intent": state["intent"]
    }

def extract_product_mentions(response: str, product_list: List[str]) -> tuple[str, List[str]]:
    """
    Extract primary and optional alternate product mentions from the chatbot response.
    Assumes first valid match is the primary, rest are alternates.
    """
    found = []
    for product in product_list:
        if re.search(rf"\b{re.escape(product)}\b", response, re.IGNORECASE):
            found.append(product)

    primary = found[0] if found else ""
    alternates = found[1:] if len(found) > 1 else []
    return primary, alternates

def update_chat_history(state: ChatbotState) -> ChatbotState:
    """
    Updates the chat history with user input, AI response, and tracks structured product mentions.
    """
    _log.info("Node: update_chat_history")
    user_query = state["user_query"]
    response = state["response"]
    chat_history = state["chat_history"]

    # Append current turn to message history
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=response))

    # Optional: extract product mentions for structured tracking
    product_list = [p.strip() for p in COMPANY_PRODUCTS.split(",") if p.strip()]
    primary_product, alternates = extract_product_mentions(response, product_list)

    _log.info(f"Primary recommendation: {primary_product}, Alternates: {alternates}")

    # You can store this metadata elsewhere if needed (e.g., CRM, DB, etc.)

    return {
        "chat_history": chat_history,
        "user_query": user_query,
        "response": response,
        "intent": state["intent"],
        "primary_product": primary_product,
        "alternate_products": alternates
    }

# -------------------- LangGraph Graph Definition --------------------

_log.info("Defining LangGraph workflow...")
workflow = StateGraph(ChatbotState)

# Add nodes
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("handle_general", handle_greeting_smalltalk_general)
workflow.add_node("retrieve_sales_info", retrieve_sales_info)
workflow.add_node("generate_sales_response", generate_sales_response)
workflow.add_node("update_history", update_chat_history)

# Set the entry point
workflow.set_entry_point("classify_intent")

# Define conditional edges based on intent classification
workflow.add_conditional_edges(
    "classify_intent",
    lambda state: state["intent"], # This function determines the next node based on 'intent'
    {
        "greeting": "handle_general",
        "small_talk": "handle_general",
        "general_query": "handle_general",
        "sales_query": "retrieve_sales_info",
        "unknown": "handle_general" # Route unknown queries to general handler
    }
)

# Define edges for the sales path
workflow.add_edge("retrieve_sales_info", "generate_sales_response")
workflow.add_edge("generate_sales_response", "update_history")

# Define edges for the general path
workflow.add_edge("handle_general", "update_history")

# Define the finish point (after history is updated)
workflow.add_edge("update_history", END)

# Compile the graph
app = workflow.compile()
_log.info("LangGraph workflow compiled successfully.")

# -------------------- Main Chatbot Runner --------------------

# Global variable to maintain session history across turns
session_chat_history: List[BaseMessage] = []

def run_arya_chatbot(user_input: str) -> str:
    """
    Runs a single turn of the Arya chatbot conversation.
    """
    global session_chat_history

    _log.info(f"User input received: {user_input}")

    # Initial state for the current turn
    initial_state = {
        "chat_history": session_chat_history,
        "user_query": user_input,
        "intent": "unknown", # Will be updated by classify_intent
        "retrieved_docs": [],
        "response": ""
    }

    try:
        # Invoke the LangGraph app
        final_state = app.invoke(initial_state)

        # Update session history for the next turn
        session_chat_history = final_state["chat_history"]

        return final_state["response"]
    except Exception as e:
        _log.error(f"Error during chatbot execution: {e}", exc_info=True)
        return f"I apologize, {CHATBOT_NAME} encountered an error. Please try again later."

# -------------------- Interactive CLI Chat Loop --------------------
if __name__ == "__main__":
    print(f"Hello! I'm {CHATBOT_NAME}, your sales assistant for {COMPANY_NAME}.")
    print("How can I help you today? (Type 'exit' to end the chat)")

    while True:
        try:
            user_message = input("\nYou: ").strip()
            if user_message.lower() == 'exit':
                print(f"{CHATBOT_NAME}: Goodbye! Have a great day.")
                break

            if not user_message:
                print(f"{CHATBOT_NAME}: Please type something.")
                continue

            response = run_arya_chatbot(user_message)
            print(f"{CHATBOT_NAME}: {response}")
        except KeyboardInterrupt:
            print(f"\n{CHATBOT_NAME}: Goodbye! Have a great day.")
            break
        except Exception as e:
            _log.critical(f"An unhandled error occurred in the CLI loop: {e}", exc_info=True)
            print(f"{CHATBOT_NAME}: I'm sorry, something unexpected went wrong. Please restart the chat.")
            break