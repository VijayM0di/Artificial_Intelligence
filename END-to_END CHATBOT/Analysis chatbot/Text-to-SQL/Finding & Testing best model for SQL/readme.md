# Text-to-SQL

This project demonstrates how to turn natural language questions into SQL queries using database schemas. It leverages a Jupyter notebook and uses Ollama to interact with language models for generating SQL queries from user input.

## Features
- Converts natural language questions to SQL queries
- Utilizes database schema information
- Integrates with Ollama for model interaction
- Jupyter notebook for interactive experimentation

## Setup Instructions

1. **Clone the repository** (or copy this folder to your machine).
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install and set up Ollama** (see [Ollama documentation](https://github.com/jmorganca/ollama) for details).

## Usage

1. Open the Jupyter notebook `testmodels.ipynb` in your preferred environment:
   ```bash
   jupyter notebook testmodels.ipynb
   ```
2. Follow the instructions in the notebook to:
   - Input a natural language question
   - Provide the relevant database schema
   - Generate and review the SQL query output

## Example
- **Input:**
  > "Show me all customers who made a purchase in the last month."
- **Output:**
  > `SELECT * FROM customers WHERE purchase_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH);`

## Notes
- Make sure Ollama is running and accessible before using the notebook.
- You may need to adjust the database schema input in the notebook to match your use case.

## License
This project is provided under the MIT License. See the LICENSE file for details. 

