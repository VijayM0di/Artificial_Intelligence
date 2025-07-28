# smolagents

**A lightweight agent framework for code‑driven tasks with language models and external tools.**

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)
3. [UseCase](#UseCase)
4. [Installation](#installation)  
5. [Quick Start](#quick-start)  
6. [Usage Example](#usage-example)  
7. [API Reference](#api-reference)  
8. [Current Limitations](#current-limitations)     
9. [License](#license)  

---

## Overview

`smolagents` is a minimalist framework that lets you combine a Hugging Face–backed language model with external tools (e.g., web search) to build “code agents” capable of answering questions, performing computations, and more.

---

## Features

- **Pluggable Tools**  
  Easily add search, file I/O, or any custom tool.  
- **Hugging Face API Integration**  
  Out‑of‑the‑box support for HF’s text models.  
- **Lightweight**  
  Minimal dependencies for rapid prototyping.

---
## UseCase

`smolagents` is being used as a query bot which retrieves data from our database to answer queries. Specifically, it is intended to:

- Interface with our database  
- Serve as a query bot
---

## Installation

```bash
pip install smolagents
```

Or clone the repo and install locally:

```bash
git clone https://github.com/your-org/smolagents.git
cd smolagents
pip install -e .
```

---

## Quick Start

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# 1. Initialize model
model = HfApiModel()

# 2. Create agent with search tool
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model
)

# 3. Run a query
result = agent.run(
    "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
)
print(result)
```

---

## Usage Example

```python
# More detailed example
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# Setup
model = HfApiModel(api_key="YOUR_HF_API_KEY")
search_tool = DuckDuckGoSearchTool(max_results=5)

agent = CodeAgent(tools=[search_tool], model=model)

# Execute
query = "Calculate the time for a cheetah at 30 m/s to cover 100 meters."
answer = agent.run(query)

# Output
# -> "At 30 m/s, it takes about 3.33 seconds to cover 100 meters."
print(answer)
```

---

## API Reference

### `HfApiModel`

- **Constructor**  
  ```python
  HfApiModel(api_key: Optional[str] = None, model_name: str = "gpt-neo-2.7B")
  ```
- **Methods**  
  - `.generate(prompt: str) → str`

### `DuckDuckGoSearchTool`

- **Constructor**  
  ```python
  DuckDuckGoSearchTool(max_results: int = 3, safe_search: bool = True)
  ```
- **Methods**  
  - `.search(query: str) → List[SearchResult]`

### `CodeAgent`

- **Constructor**  
  ```python
  CodeAgent(tools: List[Tool], model: LanguageModel)
  ```
- **Methods**  
  - `.run(prompt: str) → str`

---

## Current Limitations

- Not scalable  
- Not usable for our particular use case  
- Database connectivity is inadequate  
- Framework integration is not robust  
- Only supports tool‑based customization


---

## License

This project is licensed under the [MIT License](LICENSE).

