#!/usr/bin/env python
import sqlite3
from new_agent.crew import AgentCrew  # Ensure this import path is correct

def run():
    """
    Run the crew for query validation and text generation.
    """
    inputs = {
        'query': 'What is the positon of john doe?',  # Example query
    }
    
    try:
        # Initialize and run the crew
        crew = AgentCrew().crew()
        crew.kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


