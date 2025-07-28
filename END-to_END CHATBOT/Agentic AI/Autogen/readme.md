# Autogen Exploration

This repository documents our exploration into using the Autogen framework for intelligent task automation. While Autogen is fully functional for basic tasks, such as a simple chat interface, we encountered significant limitations when applying it to more advanced use cases.

## Overview

Autogen is designed to orchestrate tasks by managing agents and tools through OpenAI. Custom agents can leverage the OpenAI wrapper; however, this design limits the scope for more complex workflows.

## What Works

**Basic Chat Operations:**  
Autogen handles simple tasks like chatting effectively.

## Limitations of Autogen

Our experiments revealed critical limitations in the current Autogen implementation:

### Tool Routing via OpenAI

All tools are channeled through OpenAI, which restricts independent customization and functionality.

### Custom Agent Constraints

Although Autogen permits building custom agents that use the OpenAI wrapper, this integration is too restrictive without the support of new utilities (e.g., updated Ollama integration).

### Dead-End for Advanced Use Cases

Given the current limitations, we have concluded that Autogen is a dead end for scenarios requiring advanced task handling, such as generating SQL queries directly from database schemas, until further enhancements are introduced.

## Demo Script

Below is a demonstration script that we developed using Autogen in conjunction with the Ollama wrapper. This example illustrates our attempt to generate a SQL query from a user’s question:

```python
import autogen_core.models
from autogen_ext.models.ollama import OllamaChatCompletionClient
import psycopg2

# Database configuration
db_uri = "postgresql://postgres:123@localhost:5432/demodb"

# Ollama model setup
ollama_client = OllamaChatCompletionClient(model="phi4:14b")  # Choose an appropriate Ollama model

async def generate_sql_query(user_question):
    """Uses the Ollama model to generate a SQL query from the user's question."""
    prompt = f"""
    You are an advanced SQL query generator. Based on the question provided, infer the database schema and generate the corresponding SQL query.

    Question: '{user_question}'
    """
    
    response = await ollama_client.create([
        autogen_core.models.UserMessage(content=prompt, source="user")
    ])
    return response.content

async def main():
    user_question = input("Enter your question: ")
    sql_query = await generate_sql_query(user_question)
    print("Generated SQL Query:", sql_query)
    try:
        with psycopg2.connect(db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                results = cur.fetchall()
                print("Query Results:", results)
    except Exception as e:
        print("Error executing query:", e)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```
# Conclusion
The current capabilities of Autogen are sufficient for simple use cases; however, due to its inherent limitations—especially in terms of tool routing and custom agent functionality—it does not yet support advanced operations like automated SQL query generation from database input.
