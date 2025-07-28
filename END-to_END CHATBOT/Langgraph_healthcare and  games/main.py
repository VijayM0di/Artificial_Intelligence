# Travel Assistant using Google Gemini API with LangGraph
# Fixed version that properly handles LangGraph message objects

import getpass
import os
import sqlite3
import shutil
import json
import re
from datetime import date, datetime
from typing import Annotated, Optional, Union, TypedDict, List, Dict, Any
import numpy as np
import pandas as pd
import requests
import pytz

# Gemini imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# LangGraph imports - using proper message types
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Only need Gemini API key now
_set_env("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Database setup (unchanged)
db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
backup_file = "travel2.backup.sqlite"

def setup_database():
    """Setup and update database with current dates"""
    overwrite = False
    if overwrite or not os.path.exists(local_file):
        response = requests.get(db_url)
        response.raise_for_status()
        with open(local_file, "wb") as f:
            f.write(response.content)
        shutil.copy(local_file, backup_file)

    def update_dates(file):
        shutil.copy(backup_file, file)
        conn = sqlite3.connect(file)
        cursor = conn.cursor()

        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';", conn
        ).name.tolist()
        tdf = {}
        for t in tables:
            tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

        example_time = pd.to_datetime(
            tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
        ).max()
        current_time = pd.to_datetime("now").tz_localize(example_time.tz)
        time_diff = current_time - example_time

        tdf["bookings"]["book_date"] = (
            pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
            + time_diff
        )

        datetime_columns = [
            "scheduled_departure",
            "scheduled_arrival", 
            "actual_departure",
            "actual_arrival",
        ]
        for column in datetime_columns:
            tdf["flights"][column] = (
                pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
            )

        for table_name, df in tdf.items():
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        del df
        del tdf
        conn.commit()
        conn.close()
        return file

    return update_dates(local_file)

db = setup_database()

# Gemini-based vector store replacement
class GeminiVectorStoreRetriever:
    """Vector store using Gemini embeddings instead of OpenAI"""
    
    def __init__(self, docs: list, vectors: list):
        self._arr = np.array(vectors)
        self._docs = docs

    @classmethod
    def from_docs(cls, docs):
        # Use Gemini to create embeddings
        vectors = []
        for doc in docs:
            try:
                # Use Gemini's embedding API
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=doc["page_content"]
                )
                vectors.append(result['embedding'])
            except Exception as e:
                print(f"Error creating embedding: {e}")
                # Fallback: create a simple hash-based vector
                text = doc["page_content"]
                vector = [hash(text[i:i+10]) % 1000 / 1000.0 for i in range(0, min(len(text), 100), 10)]
                vector = vector + [0] * (100 - len(vector))  # Pad to fixed size
                vectors.append(vector)
        
        return cls(docs, vectors)

    def query(self, query: str, k: int = 5) -> list[dict]:
        try:
            # Get embedding for query
            query_result = genai.embed_content(
                model="models/text-embedding-004", 
                content=query
            )
            query_vector = np.array(query_result['embedding'])
            
            # Calculate similarities
            scores = query_vector @ self._arr.T
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
            
            return [
                {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
            ]
        except Exception as e:
            print(f"Error in query: {e}")
            # Fallback: return first k documents
            return self._docs[:k]

# Load FAQ data
response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
response.raise_for_status()
faq_text = response.text
docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

# Create retriever with Gemini
retriever = GeminiVectorStoreRetriever.from_docs(docs)

# State definition - FIXED to properly handle LangGraph messages
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # Proper message objects
    user_info: Optional[str]  # Current user ID

# Tool functions (converted to work with Gemini)
def lookup_policy(query: str) -> str:
    """Consult company policies using Gemini-based retrieval"""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])

def fetch_user_flight_information(passenger_id: str) -> list[dict]:
    """Fetch all tickets for the user"""
    if not passenger_id:
        raise ValueError("No passenger ID provided.")
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, 
        f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]
    cursor.close()
    conn.close()
    return results

def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    start_time: Optional[date] = None,
    end_time: Optional[date] = None,
    limit: int = 20,
) -> list[dict]:
    """Search for flights based on criteria"""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    query = "SELECT * FROM flights WHERE 1 = 1"
    params = []
    
    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)
    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)
    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(start_time)
    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(end_time)
    
    query += " LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]
    cursor.close()
    conn.close()
    return results

# Gemini-powered Assistant - FIXED to handle proper message objects
class GeminiAssistant:
    """Main assistant using Gemini instead of LangChain models"""
    
    def __init__(self):
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash",  # Use Pro for better reasoning
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        # Available tools
        self.tools = {
            "lookup_policy": lookup_policy,
            "fetch_user_flight_information": fetch_user_flight_information,
            "search_flights": search_flights,
        }
    
    def __call__(self, state: State) -> Dict[str, Any]:
        """Main assistant node function - FIXED to handle message objects"""
        messages = state.get("messages", [])
        user_info = state.get("user_info", "Unknown")
        
        # Build conversation context - FIXED to handle BaseMessage objects
        conversation_text = self._build_conversation_context(messages, user_info)
        
        # System prompt
        system_prompt = f"""You are a helpful customer support assistant for Swiss Airlines.
        
Current user: {user_info}
Current time: {datetime.now()}

You can help with:
- Flight information and changes
- Hotel and car rental bookings  
- Trip recommendations
- Company policy questions

Available tools (call by writing the function name with parameters):
- lookup_policy("query"): Search company policies
- fetch_user_flight_information("passenger_id"): Get user's flight info
- search_flights(departure_airport="CODE", arrival_airport="CODE"): Find flights

When you need to use a tool, write it exactly like: fetch_user_flight_information("{user_info}")
Be helpful, accurate, and professional.
"""

        try:
            # Generate response
            full_prompt = f"{system_prompt}\n\nConversation:\n{conversation_text}\n\nAssistant:"
            response = self.model.generate_content(full_prompt)
            
            # Check if we need to use tools
            response_text = response.text
            tool_calls = self._extract_tool_calls(response_text)
            
            if tool_calls:
                # Execute tools and generate follow-up response
                tool_results = []
                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call)
                    tool_results.append(result)
                
                # Generate final response with tool results
                tool_context = "\n".join([f"Tool {t['name']}: {t['result']}" for t in tool_results])
                final_prompt = f"{full_prompt}\n\nTool Results:\n{tool_context}\n\nFinal Response (don't mention the tools, just provide helpful answer):"
                final_response = self.model.generate_content(final_prompt)
                response_text = final_response.text
            
            # Return updated state with proper AIMessage
            new_message = AIMessage(content=response_text)
            return {"messages": [new_message]}
            
        except Exception as e:
            error_message = AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")
            return {"messages": [error_message]}
    
    def _build_conversation_context(self, messages: List[BaseMessage], user_info: str) -> str:
        """Build conversation context from message objects - FIXED"""
        context = []
        for msg in messages[-10:]:  # Last 10 messages
            # Handle different message types properly
            if isinstance(msg, HumanMessage):
                role = "Human"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            else:
                role = "System"
            
            content = msg.content if hasattr(msg, 'content') else str(msg)
            context.append(f"{role}: {content}")
        return "\n".join(context)
    
    def _extract_tool_calls(self, text: str) -> List[Dict]:
        """Extract tool calls from response text"""
        tool_calls = []
        
        # Look for patterns like "lookup_policy('search term')"
        patterns = {
            "lookup_policy": r'lookup_policy\(["\']([^"\']+)["\']\)',
            "fetch_user_flight_information": r'fetch_user_flight_information\(["\']([^"\']+)["\']\)',
            "search_flights": r'search_flights\(([^)]+)\)',
        }
        
        for tool_name, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if tool_name == "search_flights":
                    # Parse complex arguments for search_flights
                    tool_calls.append({
                        "name": tool_name,
                        "args": {"raw": match}
                    })
                else:
                    tool_calls.append({
                        "name": tool_name,
                        "args": match
                    })
        
        return tool_calls
    
    def _execute_tool(self, tool_call: Dict) -> Dict:
        """Execute a tool call"""
        tool_name = tool_call["name"]
        args = tool_call["args"]
        
        try:
            if tool_name == "lookup_policy":
                result = self.tools[tool_name](args)
            elif tool_name == "fetch_user_flight_information":
                result = self.tools[tool_name](args)
            elif tool_name == "search_flights":
                # For demo, just call with no arguments
                result = self.tools[tool_name](limit=5)
            else:
                result = f"Unknown tool: {tool_name}"
                
            return {"name": tool_name, "result": str(result)[:1000]}  # Limit result length
        except Exception as e:
            return {"name": tool_name, "result": f"Error: {str(e)}"}

# Create the LangGraph
def create_travel_assistant_graph():
    """Create and configure the LangGraph"""
    
    # Initialize components
    assistant = GeminiAssistant()
    
    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("assistant", assistant)
    
    # Add edges - simplified flow
    workflow.add_edge(START, "assistant")
    workflow.add_edge("assistant", END)
    
    # Add memory
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
   
    
    # from Ipython.display import display, Image
    # display(Image(app.get_graph().draw_mermaid_png()))
    # print("Graph created successfully!")
    return app

# Usage example - FIXED to use proper message objects
def run_travel_assistant():
    """Example of how to use the travel assistant"""
    
    app = create_travel_assistant_graph()
    
    # Configuration for user session
    config = {
        "configurable": {
            "thread_id": "user_123",
        }
    }
    
    # Example conversation - FIXED to use proper message objects
    initial_state = {
        "messages": [
            HumanMessage(content="Hi, I need help with my flight booking. Can you show me my current flights for passenger ID 1771-12345?")
        ],
        "user_info": "1771-12345"
    }
    
    try:
        # Run the graph
        result = app.invoke(initial_state, config)
        print("Assistant Response:")
        if result["messages"]:
            print(result["messages"][-1].content)
        else:
            print("No response generated")
    except Exception as e:
        print(f"Error running assistant: {e}")
        import traceback
        traceback.print_exc()

# Interactive chat function
def interactive_chat():
    """Run an interactive chat session"""
    
    app = create_travel_assistant_graph()
    
    config = {
        "configurable": {
            "thread_id": "interactive_session",
        }
    }
    
    print("Swiss Airlines Travel Assistant")
    print("==============================")
    print("Type 'quit' to exit\n")
    
    # Initialize with empty state
    state = {"messages": [], "user_info": "guest"}
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Thank you for using Swiss Airlines Travel Assistant!")
            break
        
        if not user_input:
            continue
        
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        
        try:
            # Get assistant response
            result = app.invoke(state, config)
            assistant_response = result["messages"][-1].content
            print(f"Assistant: {assistant_response}\n")
            
            # Update state with assistant response
            state = result
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    print("Travel Assistant with Gemini API - FIXED VERSION")
    print("===============================================")
    
    print("\nLangGraph Concepts:")
    print("1. State: Data flowing through the graph (now with proper message objects)")
    print("2. Nodes: Functions that process the state")  
    print("3. Edges: Connections between nodes")
    print("4. Tools: External functions the AI can call")
    print("5. Memory: Persistent conversation storage")
    
    print("\nChoose mode:")
    print("1. Run single example")
    print("2. Interactive chat")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        interactive_chat()
    else:
        print("\nRunning single example...")
        run_travel_assistant()