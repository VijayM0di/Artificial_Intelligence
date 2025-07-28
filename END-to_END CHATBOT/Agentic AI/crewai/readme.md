# Crew AI

Crew AI is an experimental project that explores agent-tool-task-based workflows using the CrewAI framework. The initial aim was to build a structured environment for intelligent automation, with a specific goal to create an SQLBot that operates solely on a database.

However, during development, we encountered several limitations in the CrewAI framework that prevented the successful execution of our goal. This repository documents that exploration, the issues encountered, and example files related to our attempt.

## Project Goal

Our primary objective was to create an **SQLBot** using CrewAI—one that could interact with and query a database without requiring external scripts or manual intervention. The bot would ideally operate within the agent-tool-task paradigm defined by CrewAI.

## Why It Didn't Work

While the concept was sound, practical limitations within CrewAI made it infeasible. Below are the core limitations that led us to pause this direction:

### 🛑 Limitations of Crew AI (at time of use)

1. **Database Input Ambiguity**  
   - Lack of clear guidance on how to provide and manage database input through the `main/crew/agents` structure.

2. **Model Utility Structure Issues**  
   - No clear or modular structure to support model utility integration or customization for specialized tasks.

3. **Flow vs. Crew Confusion**  
   - Unclear usage patterns between `flow` and `crew`, leading to confusion around:
     - How to connect a database.
     - How to use the LLM independently and inside a flow.
     - How tools are configured and their intended use cases.
     - How the TOML configuration files relate to database generation and usage.

These limitations are inherent to the current CrewAI framework (as of our usage) and blocked further development in the intended direction.

## Contents of This Repo

- Sample project structure for the failed SQLBot attempt
- Notes and configuration examples
- This README as a summary of our exploration

## Purpose of This Repo

While this repo does not contain a functioning product, we’ve made it public to share our findings, contribute to discussions around CrewAI’s limitations, and potentially help others facing similar issues.

## License

This project is shared under the [MIT License](LICENSE).
