# Travel Assistant using Google Gemini API with LangGraph and SQL Server
# Enterprise version with MSSQL integration

import getpass
import os
import json
import re
from datetime import date, datetime, timedelta
from typing import Annotated, Optional, Union, TypedDict, List, Dict, Any
import numpy as np
import pandas as pd
import requests
import pytz
from decimal import Decimal
import logging

# SQL Server imports
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# Gemini imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage

# Configuration and logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the application"""
    def __init__(self):
        self.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        self.SQL_SERVER = os.environ.get("SQL_SERVER", "localhost")
        self.SQL_DATABASE = os.environ.get("SQL_DATABASE", "TravelDB")
        self.SQL_USERNAME = os.environ.get("SQL_USERNAME")
        self.SQL_PASSWORD = os.environ.get("SQL_PASSWORD")
        self.SQL_DRIVER = os.environ.get("SQL_DRIVER", "ODBC Driver 17 for SQL Server")
        self.SQL_TRUSTED_CONNECTION = os.environ.get("SQL_TRUSTED_CONNECTION", "yes")

    def get_sql_connection_string(self):
        """Build SQL Server connection string"""
        driver_formatted = self.SQL_DRIVER.replace(' ', '+')
        if self.SQL_USERNAME and self.SQL_PASSWORD:
            # SQL Server Authentication
            return f"mssql+pyodbc://{self.SQL_USERNAME}:{self.SQL_PASSWORD}@{self.SQL_SERVER}/{self.SQL_DATABASE}?driver={driver_formatted}"
        else:
            # Windows Authentication
            return f"mssql+pyodbc://{self.SQL_SERVER}/{self.SQL_DATABASE}?driver={driver_formatted}&trusted_connection={self.SQL_TRUSTED_CONNECTION}"

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

def setup_environment():
    """Setup required environment variables if not already set."""
    _set_env("GEMINI_API_KEY")

    print("\nSQL Server Configuration (press Enter to use defaults):")
    if not os.environ.get("SQL_SERVER"):
        os.environ["SQL_SERVER"] = input("SQL Server instance (default: localhost): ") or "localhost"

    if not os.environ.get("SQL_DATABASE"):
        os.environ["SQL_DATABASE"] = input("Database name (default: TravelDB): ") or "TravelDB"

    auth_type = input("Use Windows Authentication? (y/n, default: y): ").lower()
    if auth_type.startswith('n'):
        _set_env("SQL_USERNAME")
        _set_env("SQL_PASSWORD")
        os.environ["SQL_TRUSTED_CONNECTION"] = "no"
    else:
        os.environ["SQL_TRUSTED_CONNECTION"] = "yes"


class SQLServerManager:
    """SQL Server database manager"""

    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        self.connection_string = config.get_sql_connection_string()
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with connection pooling"""
        try:
            self.engine = create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("SQL Server connection established successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to SQL Server: {e}")
            logger.error(f"Connection string used: {self.connection_string}")
            raise

    def create_database_schema(self):
        """Create database schema for travel management system if tables don't exist."""
        # The schema creation SQL is idempotent using `IF NOT EXISTS`
        schema_sql = """
        -- Airlines table
        IF OBJECT_ID('dbo.airlines', 'U') IS NULL
        CREATE TABLE airlines (
            airline_code VARCHAR(3) PRIMARY KEY,
            airline_name NVARCHAR(100) NOT NULL,
            country NVARCHAR(50),
            created_date DATETIME2 DEFAULT GETDATE()
        );

        -- Airports table
        IF OBJECT_ID('dbo.airports', 'U') IS NULL
        CREATE TABLE airports (
            airport_code VARCHAR(3) PRIMARY KEY,
            airport_name NVARCHAR(200) NOT NULL,
            city NVARCHAR(100),
            country NVARCHAR(100),
            timezone NVARCHAR(50),
            latitude DECIMAL(10,8),
            longitude DECIMAL(11,8),
            created_date DATETIME2 DEFAULT GETDATE()
        );

        -- Aircraft table
        IF OBJECT_ID('dbo.aircraft', 'U') IS NULL
        CREATE TABLE aircraft (
            aircraft_code VARCHAR(10) PRIMARY KEY,
            model NVARCHAR(100) NOT NULL,
            range_km INT,
            capacity INT,
            created_date DATETIME2 DEFAULT GETDATE()
        );

        -- Passengers table
        IF OBJECT_ID('dbo.passengers', 'U') IS NULL
        CREATE TABLE passengers (
            passenger_id NVARCHAR(20) PRIMARY KEY,
            passenger_name NVARCHAR(200) NOT NULL,
            email NVARCHAR(100),
            phone NVARCHAR(20),
            contact_data NVARCHAR(MAX), -- JSON data
            created_date DATETIME2 DEFAULT GETDATE(),
            updated_date DATETIME2 DEFAULT GETDATE()
        );

        -- Bookings table
        IF OBJECT_ID('dbo.bookings', 'U') IS NULL
        CREATE TABLE bookings (
            book_ref VARCHAR(10) PRIMARY KEY,
            book_date DATETIME2 NOT NULL,
            total_amount DECIMAL(10,2),
            passenger_id NVARCHAR(20),
            status NVARCHAR(20) DEFAULT 'confirmed',
            created_date DATETIME2 DEFAULT GETDATE(),
            FOREIGN KEY (passenger_id) REFERENCES passengers(passenger_id)
        );

        -- Flights table
        IF OBJECT_ID('dbo.flights', 'U') IS NULL
        CREATE TABLE flights (
            flight_id INT IDENTITY(1,1) PRIMARY KEY,
            flight_no VARCHAR(10) NOT NULL,
            scheduled_departure DATETIME2 NOT NULL,
            scheduled_arrival DATETIME2 NOT NULL,
            departure_airport VARCHAR(3) NOT NULL,
            arrival_airport VARCHAR(3) NOT NULL,
            status NVARCHAR(20) DEFAULT 'scheduled',
            aircraft_code VARCHAR(10),
            actual_departure DATETIME2,
            actual_arrival DATETIME2,
            created_date DATETIME2 DEFAULT GETDATE(),
            FOREIGN KEY (departure_airport) REFERENCES airports(airport_code),
            FOREIGN KEY (arrival_airport) REFERENCES airports(airport_code),
            FOREIGN KEY (aircraft_code) REFERENCES aircraft(aircraft_code)
        );

        -- Tickets table
        IF OBJECT_ID('dbo.tickets', 'U') IS NULL
        CREATE TABLE tickets (
            ticket_no VARCHAR(15) PRIMARY KEY,
            book_ref VARCHAR(10) NOT NULL,
            passenger_id NVARCHAR(20) NOT NULL,
            passenger_name NVARCHAR(200),
            contact_data NVARCHAR(MAX), -- JSON data
            created_date DATETIME2 DEFAULT GETDATE(),
            FOREIGN KEY (book_ref) REFERENCES bookings(book_ref),
            FOREIGN KEY (passenger_id) REFERENCES passengers(passenger_id)
        );

        -- Ticket Flights (many-to-many relationship)
        IF OBJECT_ID('dbo.ticket_flights', 'U') IS NULL
        CREATE TABLE ticket_flights (
            ticket_no VARCHAR(15),
            flight_id INT,
            fare_conditions NVARCHAR(20) DEFAULT 'Economy',
            amount DECIMAL(10,2),
            PRIMARY KEY (ticket_no, flight_id),
            FOREIGN KEY (ticket_no) REFERENCES tickets(ticket_no) ON DELETE CASCADE,
            FOREIGN KEY (flight_id) REFERENCES flights(flight_id)
        );

        -- Boarding passes table
        IF OBJECT_ID('dbo.boarding_passes', 'U') IS NULL
        CREATE TABLE boarding_passes (
            ticket_no VARCHAR(15),
            flight_id INT,
            boarding_no INT,
            seat_no VARCHAR(4),
            issued_date DATETIME2 DEFAULT GETDATE(),
            PRIMARY KEY (ticket_no, flight_id),
            FOREIGN KEY (ticket_no, flight_id) REFERENCES ticket_flights(ticket_no, flight_id) ON DELETE CASCADE
        );

        -- Hotels table (additional feature)
        IF OBJECT_ID('dbo.hotels', 'U') IS NULL
        CREATE TABLE hotels (
            hotel_id INT IDENTITY(1,1) PRIMARY KEY,
            hotel_name NVARCHAR(200) NOT NULL,
            city NVARCHAR(100),
            country NVARCHAR(100),
            rating DECIMAL(2,1),
            address NVARCHAR(500),
            phone NVARCHAR(20),
            email NVARCHAR(100),
            created_date DATETIME2 DEFAULT GETDATE()
        );

        -- Hotel Bookings table
        IF OBJECT_ID('dbo.hotel_bookings', 'U') IS NULL
        CREATE TABLE hotel_bookings (
            booking_id INT IDENTITY(1,1) PRIMARY KEY,
            passenger_id NVARCHAR(20),
            hotel_id INT,
            check_in_date DATE,
            check_out_date DATE,
            room_type NVARCHAR(50),
            total_amount DECIMAL(10,2),
            status NVARCHAR(20) DEFAULT 'confirmed',
            created_date DATETIME2 DEFAULT GETDATE(),
            FOREIGN KEY (passenger_id) REFERENCES passengers(passenger_id),
            FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id)
        );
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(text(schema_sql))
            logger.info("Database schema verified/created successfully.")
        except Exception as e:
            logger.error(f"Error creating database schema: {e}")
            raise

    def insert_sample_data(self):
        """Insert sample data for testing if it doesn't already exist."""
        # This approach is slow but safe for demonstration.
        # A better production approach would be a single script with existence checks.
        data_inserts = [
            ("airlines", "airline_code", "'LX'", "INSERT INTO airlines VALUES ('LX', 'Swiss International Air Lines', 'Switzerland', GETDATE());"),
            ("airlines", "airline_code", "'LH'", "INSERT INTO airlines VALUES ('LH', 'Lufthansa', 'Germany', GETDATE());"),
            ("airports", "airport_code", "'ZUR'", "INSERT INTO airports VALUES ('ZUR', 'Zurich Airport', 'Zurich', 'Switzerland', 'Europe/Zurich', 47.4647, 8.5492, GETDATE());"),
            ("airports", "airport_code", "'FRA'", "INSERT INTO airports VALUES ('FRA', 'Frankfurt Airport', 'Frankfurt', 'Germany', 'Europe/Berlin', 50.0379, 8.5622, GETDATE());"),
            ("airports", "airport_code", "'JFK'", "INSERT INTO airports VALUES ('JFK', 'John F. Kennedy International Airport', 'New York', 'USA', 'America/New_York', 40.6413, -73.7781, GETDATE());"),
            ("aircraft", "aircraft_code", "'A320'", "INSERT INTO aircraft VALUES ('A320', 'Airbus A320', 6150, 180, GETDATE());"),
            ("aircraft", "aircraft_code", "'B777'", "INSERT INTO aircraft VALUES ('B777', 'Boeing 777', 9700, 350, GETDATE());"),
            ("passengers", "passenger_id", "'1771-12345'", "INSERT INTO passengers VALUES ('1771-12345', 'John Doe', 'john.doe@email.com', '+1-555-0123', '{\"preferences\": {\"seat\": \"window\", \"meal\": \"vegetarian\"}}', GETDATE(), GETDATE());"),
            ("bookings", "book_ref", "'ABC123'", "INSERT INTO bookings VALUES ('ABC123', GETDATE(), 1200.00, '1771-12345', 'confirmed', GETDATE());"),
            ("hotels", "hotel_name", "'Zurich Grand Hotel'", "INSERT INTO hotels (hotel_name, city, country, rating, address, phone, email) VALUES ('Zurich Grand Hotel', 'Zurich', 'Switzerland', 4.5, '123 Main St, Zurich', '+41-44-1234567', 'info@zurichgrand.com');"),
        ]
        
        try:
            with self.engine.begin() as conn:
                for table, key_col, key_val, insert_sql in data_inserts:
                    check_query = text(f"SELECT 1 FROM {table} WHERE {key_col} = {key_val}")
                    exists = conn.execute(check_query).scalar()
                    if not exists:
                        conn.execute(text(insert_sql))
            logger.info("Sample data verified/inserted successfully.")
        except Exception as e:
            logger.error(f"Error inserting sample data: {e}")
            raise

    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute query and return pandas DataFrame."""
        try:
            with self.engine.connect() as conn:
                return pd.read_sql_query(text(query), conn, params=params)
        except Exception as e:
            logger.error(f"Error executing query: {query} with params {params}. Error: {e}")
            raise


# Enhanced tool functions for SQL Server
def lookup_policy(query: str, retriever) -> str:
    """Consult company policies and FAQs to answer customer questions."""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])

def fetch_user_flight_information(passenger_id: str, db_manager: SQLServerManager) -> str:
    """Fetch all flight tickets for a user given their passenger ID from the SQL Server database."""
    if not passenger_id:
        raise ValueError("No passenger ID provided.")
    query = """
    SELECT 
        t.ticket_no, t.book_ref, f.flight_id, f.flight_no, f.departure_airport, 
        da.city as departure_city, f.arrival_airport, aa.city as arrival_city,
        f.scheduled_departure, f.scheduled_arrival, f.status as flight_status,
        COALESCE(bp.seat_no, 'Not assigned') as seat_no, tf.fare_conditions, tf.amount
    FROM tickets t
    JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
    JOIN flights f ON tf.flight_id = f.flight_id
    JOIN airports da ON f.departure_airport = da.airport_code
    JOIN airports aa ON f.arrival_airport = aa.airport_code
    LEFT JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE t.passenger_id = :passenger_id
    ORDER BY f.scheduled_departure
    """
    result_df = db_manager.execute_query(query, {"passenger_id": passenger_id})
    if result_df.empty:
        return "No flight information found for this passenger."
    return result_df.to_json(orient='records', date_format='iso')

def search_flights(db_manager: SQLServerManager,
                   departure_airport: Optional[str] = None,
                   arrival_airport: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   limit: int = 20) -> str:
    """Search for flights in the SQL Server database based on the provided criteria."""
    query = "SELECT TOP (:limit) f.*, da.city as departure_city, aa.city as arrival_city FROM flights f " \
            "JOIN airports da ON f.departure_airport = da.airport_code " \
            "JOIN airports aa ON f.arrival_airport = aa.airport_code WHERE 1=1"
    params = {"limit": limit}
    if departure_airport:
        query += " AND f.departure_airport = :departure_airport"
        params["departure_airport"] = departure_airport
    if arrival_airport:
        query += " AND f.arrival_airport = :arrival_airport"
        params["arrival_airport"] = arrival_airport
    if start_date:
        query += " AND f.scheduled_departure >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND f.scheduled_departure <= :end_date"
        params["end_date"] = end_date
    query += " ORDER BY f.scheduled_departure"
    
    result_df = db_manager.execute_query(query, params)
    if result_df.empty:
        return "No flights found matching the criteria."
    return result_df.to_json(orient='records', date_format='iso')

def search_hotels(db_manager: SQLServerManager, city: Optional[str] = None,
                  min_rating: Optional[float] = None, limit: int = 10) -> str:
    """Search for hotels in the database based on city and minimum rating."""
    query = "SELECT TOP (:limit) * FROM hotels WHERE 1=1"
    params = {"limit": limit}
    if city:
        query += " AND city LIKE :city"
        params["city"] = f"%{city}%"
    if min_rating:
        query += " AND rating >= :min_rating"
        params["min_rating"] = min_rating
    query += " ORDER BY rating DESC"

    result_df = db_manager.execute_query(query, params)
    if result_df.empty:
        return "No hotels found matching the criteria."
    return result_df.to_json(orient='records')

# Gemini Vector Store
class GeminiVectorStoreRetriever:
    """Vector store using Gemini embeddings."""
    def __init__(self, docs: list, vectors: list):
        self._arr = np.array(vectors)
        self._docs = docs

    @classmethod
    def from_docs(cls, docs):
        logger.info("Generating embeddings for documents...")
        embeddings = genai.embed_content(
            model="models/text-embedding-004",
            content=[doc["page_content"] for doc in docs],
            task_type="RETRIEVAL_DOCUMENT"
        )
        logger.info("Embeddings generated.")
        return cls(docs, embeddings['embedding'])

    def query(self, query: str, k: int = 5) -> List[Dict]:
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
        
        scores = np.dot(query_embedding, self._arr.T)
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        
        return [{**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted]


# State definition
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_info: str

# Gemini Assistant with Tool Calling
class GeminiAssistant:
    """The assistant model that uses Gemini's tool-calling capability."""
    def __init__(self, db_manager: SQLServerManager, retriever):
        self.db_manager = db_manager
        self.retriever = retriever
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            tools=[
                lambda query: lookup_policy(query, self.retriever),
                lambda passenger_id: fetch_user_flight_information(passenger_id, self.db_manager),
                lambda **kwargs: search_flights(self.db_manager, **kwargs),
                lambda **kwargs: search_hotels(self.db_manager, **kwargs),
            ],
            system_instruction="""You are a helpful and friendly travel assistant for Swiss Airlines.
            Your user is: {user_info}.
            Current time: {time}.
            You have access to a SQL Server database with flight, booking, and hotel information.
            Use your tools to answer user questions about their flights, search for new flights or hotels, and look up company policies.
            When replying to the user, summarize the results from the tool calls in a clear, easy-to-read format.
            Do not just return the raw JSON data. For example, if a user asks for their flights, list them out with key details.
            If a search returns no results, inform the user clearly.
            Be polite and always ask if there is anything else you can help with."""
        )

    def __call__(self, state: State) -> Dict[str, Any]:
        """Main assistant logic."""
        user_info = state["user_info"]
        system_instruction = self.model.system_instruction.format(
            user_info=user_info, time=datetime.now().isoformat()
        )
        
        chat = self.model.start_chat(history=state['messages'])
        latest_user_message = state['messages'][-1].content
        
        try:
            response = chat.send_message(latest_user_message)
            return {"messages": [response.candidates[0].content]}
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            error_message = AIMessage(content=f"Sorry, I encountered an error: {e}")
            return {"messages": [error_message]}


# Graph definition
class Agent:
    def __init__(self, assistant_runnable):
        self.runnable = assistant_runnable

    def __call__(self, state: State):
        result = self.runnable.invoke(state)
        return {"messages": result["messages"]}

def create_travel_assistant_graph(config: Config):
    """Create and configure the LangGraph with SQL Server and Gemini."""
    db_manager = SQLServerManager(config)
    
    print("Setting up database schema and data...")
    db_manager.create_database_schema()
    db_manager.insert_sample_data()

    print("Setting up FAQ retriever...")
    try:
        response = requests.get("https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md")
        response.raise_for_status()
        faq_text = response.text
        docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text) if txt.strip()]
        retriever = GeminiVectorStoreRetriever.from_docs(docs)
    except Exception as e:
        logger.error(f"Could not load FAQ data: {e}. Using dummy data.")
        docs = [{"page_content": "Swiss Airlines FAQ - Please contact customer service at 1-800-SWISS-AIR for assistance."}]
        retriever = GeminiVectorStoreRetriever.from_docs(docs)
        
    assistant_runnable = GeminiAssistant(db_manager, retriever)
    agent = Agent(assistant_runnable)

    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    workflow.set_entry_point("agent")
    workflow.set_finish_point("agent")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

# Main execution block
def main():
    """Main function to run the travel assistant."""
    print("--- Travel Assistant with Gemini, LangGraph, and SQL Server ---")
    
    # Setup environment variables interactively
    setup_environment()
    
    try:
        # Create config and the graph
        config = Config()
        graph = create_travel_assistant_graph(config)
    except Exception as e:
        print(f"\nFATAL: Could not initialize the application. Please check your configuration. Error: {e}")
        return

    # Set up conversation
    passenger_id = "1771-12345" # Example passenger
    config_thread = {"configurable": {"thread_id": passenger_id}}
    
    # Set initial user info in the state
    initial_state = {
        "messages": [],
        "user_info": f"Passenger ID: {passenger_id}, Name: John Doe"
    }
    graph.update_state(config_thread, initial_state)

    print(f"\nWelcome, John Doe (Passenger ID: {passenger_id}). How can I help you today?")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        events = graph.stream({"messages": [HumanMessage(content=user_input)]}, config_thread, stream_mode="values")
        for event in events:
            # The 'agent' node is the only one, so we can directly access its output
            if "agent" in event:
                latest_message = event["agent"]["messages"][-1]
                if isinstance(latest_message.content, str):
                    print(f"Assistant: {latest_message.content}")
                else: # Tool calls are structured
                    print(f"Assistant: {latest_message}")


if __name__ == "__main__":
    main()

