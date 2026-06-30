# 🧬 Medical Research Assistant (RAG)

A fully local Retrieval-Augmented Generation (RAG) application for analyzing medical research papers.

## Features
-  Upload multiple PDF research papers
-  Ask questions about papers
-  Generate structured summaries
-  Compare multiple studies
-  Generate literature reviews
-  Extract key findings
-  Fully local using Ollama + LangChain + FAISS

## Tech Stack
- Python
- Streamlit
- LangChain
- Ollama (Llama 3.2)
- FAISS
- Sentence Transformers

## Installation

```bash
git clone <repo-url>
cd medical_assis
pip install -r requirements.txt
```

Start Ollama:

```bash
ollama pull llama3.2
ollama run llama3.2
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
medical_assis/
│
├── app.py
├── requirements.txt
├── src/
│   ├── pdf_loader.py
│   ├── vector_store.py
│   └── rag_engine.py
├── data/
└── README.md
```