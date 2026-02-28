# MKT3434 – Introduction to Machine Learning (2026)

**Mechatronics Engineering Department, Yıldız Technical University**  
**Instructor:** Ertugrul Bayraktar

---

## Term Project: RAG & MCP-Based Document Interaction with LLM Monitoring

### Project Overview

In this term project you will build an intelligent document-interaction system that extends the provided starter application.  The system combines three modern AI techniques:

| Technique | Role |
|-----------|------|
| **RAG** (Retrieval-Augmented Generation) | Query large document corpora accurately |
| **MCP** (Model Context Protocol) | Connect the LLM to external tools and APIs |
| **LLM Monitoring** | Log and evaluate every model interaction |

You will work with document collections **exceeding 500 pages** that span a variety of formats and topics:

| Format | Example topics |
|--------|----------------|
| TXT, PDF, DOCX | Health, Engineering, Science |
| JSON, CSV | Sports statistics, Political data |
| Markdown | Technical documentation, Education |

---

## Repository Structure

```
MKT3434_2026/
├── app.py            ← Starter PySide6 GUI (extend this)
├── rag_engine.py     ← RAG document loader & retrieval chain
├── mcp_client.py     ← MCP tool registration & invocation
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---

## Getting Started

### Prerequisites

- Python **3.10** or higher
- `pip` package manager
- *(Optional)* [Ollama](https://ollama.com) for free, local model inference

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/bayraktare/MKT3434_2026.git
cd MKT3434_2026

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```

> **No API key?**  Select any `ollama/…` model from the dropdown and leave the API Key field blank.  Make sure Ollama is running locally (`ollama serve`).

---

## Application Features

The starter GUI (`app.py`) provides:

| Panel | Features |
|-------|----------|
| **Model Selection** | Dropdown with GPT-4o, Claude 3.5, Ollama (llama3.2, mistral, phi3) |
| **Source Selection** | Add / remove document files; supported formats: TXT, PDF, DOCX, JSON, CSV, MD |
| **Load & Index** | Chunk documents and build a ChromaDB vector store |
| **Prompt Input** | Free-text query area with Send / Clear buttons |
| **Response Output** | Timestamped answers with source attribution |
| **MCP Tool Panel** | Lists registered MCP tools available to the model |
| **LLM Monitor** | Live log of every query, model, and response length; full history in `app.log` |

### Application Layout

```
┌─────────────────────────────────────────────────────────────┐
│       MKT3434 – Introduction to Machine Learning            │
├───────────────────────┬─────────────────────────────────────┤
│  Model Selection      │  Source Selection                   │
│  • LLM dropdown       │  • File list (TXT/PDF/DOCX/JSON/CSV)│
│  • API key field      │  • Add / Remove / Load & Index      │
│  • MCP tools list     │                                     │
├───────────────────────┴─────────────────────────────────────┤
│  Prompt Input                          [Clear]  [Send Query]│
├─────────────────────────────────────────────────────────────┤
│  Response Output  (timestamped, scrollable)                 │
├─────────────────────────────────────────────────────────────┤
│  LLM Interaction Monitor  (live log → app.log)              │
├─────────────────────────────────────────────────────────────┤
│  Status bar                              [progress spinner] │
└─────────────────────────────────────────────────────────────┘
```

---

## System Architecture

```
┌─────────────────────────────────────┐
│         PySide6 GUI  (app.py)       │
│   QueryWorker thread (non-blocking) │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      RAG Engine  (rag_engine.py)    │
│  Document Loader → Text Splitter    │
│  → Embeddings → ChromaDB           │
│  → RetrievalQA Chain → Answer      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     MCP Client  (mcp_client.py)     │
│  Tool registry & invocation         │
│  Extensible to remote MCP servers   │
└─────────────────────────────────────┘
```

---

## Project Requirements

Students must extend the starter code to satisfy **all** of the following:

1. **RAG corpus** – Index a document collection totalling at least 500 pages, covering at least 3 different topics.
2. **Multi-format loading** – Use at least 3 different file formats (e.g. PDF + DOCX + JSON).
3. **MCP integration** – Register at least 1 new MCP tool that the LLM can call (e.g. web search, database query, calculator).
4. **Response monitoring** – Implement a quality-metric logging system (e.g. relevance score, answer length, latency).
5. **GUI enhancements** – Add at least 2 new features to the GUI beyond what is already provided.
6. **Documentation** – Update `README.md` with your own setup instructions and sample outputs.

---

## Grading Criteria

| Criteria | Points |
|----------|--------|
| Working RAG implementation with 500+ page corpus | 30 |
| MCP integration with at least 1 external tool | 20 |
| LLM response monitoring with quality metrics | 20 |
| GUI enhancements (2+ new features) | 15 |
| Documentation and code quality | 15 |
| **Total** | **100** |

---

## Submission Guidelines

### Deadline: **Week 12**

> ⚠️ **Submissions that do not follow the naming convention below will NOT be checked.**

### Naming Convention

Name your submission file using **your student ID number**:

```
your_student_id_number.zip
```

**Example:** `19036154.zip`

### Contents of Your ZIP

```
19036154.zip
├── app.py              (your extended GUI)
├── rag_engine.py       (your RAG implementation)
├── mcp_client.py       (your MCP integration)
├── requirements.txt    (all dependencies)
├── README.md           (your documentation)
├── data/               (sample documents used for testing)
└── logs/               (sample app.log showing monitored interactions)
```

### Submission Rules

- ✅ ZIP file named **exactly** as your student ID number (e.g., `19036154.zip`)
- ✅ Include all source files and at least 5 sample queries with logged responses
- ✅ Virtual environments (`venv/`) and vector stores (`.chroma_db/`) must **not** be included
- ❌ **Duplicate submissions will not be accepted**
- ❌ Submissions with incorrect naming will **not** be checked
- ❌ Late submissions after Week 12 will not be accepted

---

## License

This starter code is provided for educational purposes for students enrolled in MKT3434 at Yıldız Technical University.  All rights reserved by Ertugrul Bayraktar.
