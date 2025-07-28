# Langgraph Healthcare and Games

This project contains a collection of healthcare and game-related tools, bots, and flowcharts, including:
- Healthcare chatbot flows
- Travel assistant flows
- Word and number games
- Jupyter notebooks for customer support and experimentation
- Docker Compose setup for multi-service orchestration

## Features
- Healthcare and travel assistant flowcharts (Mermaid diagrams)
- Multiple Python scripts for bots and assistants
- Jupyter notebooks for interactive demos
- Word and number game implementation
- Docker Compose for easy multi-service setup

## Setup Instructions

1. **Clone the repository** (or copy this folder to your machine).
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Use Docker Compose:
   ```bash
   docker-compose up
   ```

## Usage
- Run Python scripts directly, e.g.:
  ```bash
  python main.py
  ```
- Open Jupyter notebooks for interactive demos:
  ```bash
  jupyter notebook hello.ipynb
  ```
- View flowcharts using a Mermaid viewer or compatible Markdown editor.

## Notes
- Large files such as `travel2.sqlite` and `travel2.backup.sqlite` are included for demonstration but should not be committed to version control. Add them to `.gitignore` if not needed.
- The `.conda/` directory and other environment files should also be excluded from git.

## License
This project is provided under the MIT License. See the LICENSE file for details. 