from crewai_tools import NL2SQLTool

# psycopg2 was installed to run this example with PostgreSQL
nl2sql = NL2SQLTool(db_uri="postgresql://postgres@localhost:5432/data")

@agent
def sql_generator(self) -> Agent:
    return Agent(
        config=self.agents_config["sql_generator"],
        allow_delegation=False,
        tools=[nl2sql]
    )