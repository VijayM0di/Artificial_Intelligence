# Salesbot (Arya Chatbot)

![MIT License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

**AI Sante Salesbot** is an AI-powered sales assistant designed for the pharmaceutical and healthcare industry. It leverages advanced language models and document retrieval to answer queries, provide product information, and assist sales teams using your own knowledge base.

---

## Features

- **Document-based Q&A:** Answers questions using your own PDF documents and knowledge base.
- **Intent Classification:** Detects user intent (greeting, small talk, sales, general query).
- **Contextual Retrieval:** Uses ChromaDB for fast, relevant document search.
- **Human-like Responses:** Generates WhatsApp-friendly, concise, and professional replies.
- **Product Promotion:** Suggests products and provides calls-to-action based on user queries.
- **Extensible:** Easily add new documents or update the knowledge base.

---

## Project Structure

Salesbot/
│
├── src/ # Main source code (arya_chat_v1.py)
├── data/
│ └── pdfs/ # PDF documents (not tracked in git)
├── chroma_db/ # ChromaDB vector store (not tracked in git)
├── requirements.txt # Python dependencies
├── README.md # Project overview and instructions
├── .gitignore # Files/folders to ignore in git
└── LICENSE # License for your code



---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/VijayM0di/Salesbot.git
cd Salesbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Environment

- Create a `.env` file in the root directory with your API keys (e.g., `GOOGLE_API_KEY` for LLMs).

### 4. Add Your Documents

- Place your PDF files in `data/pdfs/`.
- (Optional) Use your own scripts to process and embed these documents into ChromaDB.

### 5. Generate ChromaDB

- If you don’t have a prebuilt `chroma_db/`, run your embedding script to generate it from the PDFs.
- **Note:** `chroma_db/` and `data/pdfs/` are not tracked in git due to size.

### 6. Run the Bot

```bash
python src/arya_chat_v1.py
```

---

## Usage

- Interact with the bot via the command line or integrate it into your application.
- The bot will classify user queries, retrieve relevant information from your documents, and generate a response.

---

## Customization

- **Add/Update Documents:** Place new PDFs in `data/pdfs/` and regenerate the ChromaDB.
- **Change Model:** Update the LLM or embedding model in `src/arya_chat_v1.py`.
- **Extend Functionality:** Add new intent handlers or utilities as needed.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Credits

Developed by [Your Name/Team] for AI Sante.

---

## Contact

- vijaymodi2002@gmail.com
