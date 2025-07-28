from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import NL2SQLTool

@CrewBase
class AgentCrew:
    """Agent crew for handling SQL tasks"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    nl2sql = NL2SQLTool(db_uri="postgresql://postgres:123@localhost:5432/demodb")

    @agent
    def sql_generator(self) -> Agent:
        """Creates an SQL Generator agent."""
        return Agent(
            config=self.agents_config['sql_generator'],
            verbose=True,
            tools=[self.nl2sql]
        )

    @agent
    def sql_validator(self) -> Agent:
        """Creates an SQL Validator agent."""
        return Agent(
            config=self.agents_config['sql_validator'],
            verbose=True
        )

    @agent
    def sql_to_text(self) -> Agent:
        """Creates an SQL to Text Generator agent."""
        return Agent(
            config=self.agents_config['sql_to_text'],
            verbose=True
        )

    @task
    def generate_sql_task(self) -> Task:
        """Defines the task for generating SQL queries."""
        return Task(
            config=self.tasks_config['generate_sql_task'],
        )

    @task
    def validate_sql_task(self) -> Task:
        """Defines the task for validating SQL queries."""
        return Task(
            config=self.tasks_config['validate_sql_task'],
        )

    @task
    def sql_to_text_task(self) -> Task:
        """Defines the task for converting SQL results to text."""
        return Task(
            config=self.tasks_config['sql_to_text_task'],
            output_file='output_report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Agent crew."""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
