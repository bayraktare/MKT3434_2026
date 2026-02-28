# MKT3454 — Introduction to Machine Learning  
## Term Project: RAG & MCP Explorer

**Yıldız Technical University · Mechatronics Engineering Department**

---

```
 ██████╗ █████╗  ██████╗      ██████╗  █████╗ ██╗  ██╗
 ██╔══██╗██╔══██╗██╔════╝     ██╔══██╗██╔══██╗██║ ██╔╝
 ██████╔╝███████║██║  ███╗    ██████╔╝███████║█████╔╝ 
 ██╔══██╗██╔══██║██║   ██║    ██╔══██╗██╔══██║██╔═██╗ 
 ██║  ██║██║  ██║╚██████╔╝    ██║  ██║██║  ██║██║  ██╗
 ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
      Retrieval-Augmented Generation  +  MCP Explorer
```

---

## 📌 Overview

This repository contains the **starter code** for the MKT3454 term project. You will build a complete RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol) application on top of the provided graphical interface.

The project tests your ability to:

- Index and query **large, heterogeneous document collections** (500+ pages, TXT / PDF / DOCX / JSON / CSV / HTML)
- Implement a **vector-store-backed retrieval pipeline** (chunking → embedding → similarity search)
- Integrate with **multiple LLM providers** (Anthropic Claude, OpenAI GPT, Google Gemini, Ollama)
- Connect to an **MCP server** and expose tool-calling to the model
- **Monitor and log** every LLM interaction (latency, token usage, metadata)

---

## 🗂 Repository Structure

```
mkt3454_project/
│
├── main.py             ← GUI (provided – do not modify the layout/signals)
├── rag_pipeline.py     ← RAG pipeline  ← YOU IMPLEMENT THIS
├── llm_monitor.py      ← LLM monitor   ← YOU IMPLEMENT THIS
├── mcp_handler.py      ← MCP handler   ← YOU IMPLEMENT THIS
│
├── requirements.txt    ← All dependencies
├── .gitignore
├── README.md           ← This file
│
└── data/               ← Put your test documents here (not committed to git)
    ├── sample.txt
    ├── sample.pdf
    └── ...
```

> **Rule:** Do **not** rename the public methods in `rag_pipeline.py`, `llm_monitor.py`, or `mcp_handler.py`. The GUI calls them by name.

---

## ⚙️ Setup

### 1 — Clone the repository

```bash
git clone https://github.com/<instructor-handle>/mkt3454-term-project.git
cd mkt3454-term-project
```

### 2 — Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> Some packages (e.g. `faiss-cpu`) may require a C++ build toolchain.  
> On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if you encounter errors.

### 4 — Set API keys

Create a `.env` file in the project root (it is git-ignored):

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...          # optional – for Gemini
```

Or enter your key directly in the GUI's **API KEY** field at runtime.

### 5 — Run the application

```bash
python main.py
```

---

## 🖥️ GUI Walkthrough

| Area | Description |
|------|-------------|
| **Model** | Select the LLM to use for generation. |
| **MCP Server** | Toggle to connect / disconnect an MCP server. |
| **Data Sources** | Add individual files or entire folders. Supports `.txt`, `.pdf`, `.docx`, `.json`, `.csv`, `.md`, `.html`. |
| **Retrieval Settings** | Chunk size and Top-K for the similarity search. |
| **Build / Refresh Index** | Loads all added documents, splits them into chunks, and builds the vector index. |
| **Prompt** | Type your question and click **Send Query**. |
| **Response** | Streamed model output. |
| **Retrieved Context Chunks** | The document passages injected into the model's context. |
| **Monitor Tab** | Per-query metadata: model, elapsed time, token counts. |
| **Session Log Tab** | Full timestamped log of all application events. |

---

## 📝 What You Must Implement

### `rag_pipeline.py`

| Method | Points |
|--------|--------|
| `build_index()` – load all file types | 20 |
| `build_index()` – chunking & embedding | 15 |
| `build_index()` – vector store (Chroma / FAISS) | 15 |
| `query()` – retrieval & prompt augmentation | 20 |
| `query()` – LLM call (streaming supported) | 20 |

### `llm_monitor.py`

| Method | Points |
|--------|--------|
| `summary()` – full statistics | 5 |
| Any additional analytics (faithfulness, topic classification, etc.) | Bonus 5 |

### `mcp_handler.py`

| Method | Points |
|--------|--------|
| `connect()` + `list_tools()` | 3 |
| `call_tool()` | 2 |

**Total: 100 points + 5 bonus**

---

## 📚 Recommended Libraries

```python
# Document loading
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, JSONLoader, CSVLoader
)

# Text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings   # free, local
from langchain_openai import OpenAIEmbeddings                       # requires key

# Vector stores
from langchain_community.vectorstores import Chroma                 # recommended
from langchain_community.vectorstores import FAISS                  # alternative

# LLM clients
import anthropic          # Claude
import openai             # GPT
import google.generativeai as genai   # Gemini

# MCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
```

---

## 💡 Hints

### Loading documents

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("my_report.pdf")
pages  = loader.load()   # list of Document objects
```

### Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=51)
chunks   = splitter.split_documents(pages)
```

### Building a Chroma vector store

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings  import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb   = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
```

### Querying

```python
docs    = vectordb.similarity_search(prompt, k=5)
context = "\n\n".join(d.page_content for d in docs)
```

### Streaming with Claude

```python
import anthropic

client = anthropic.Anthropic()
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}],
) as stream:
    for text in stream.text_stream:
        stream_callback(text)   # send to GUI
full_response = stream.get_final_message()
```

---

## 📦 Submission Instructions

> ⚠️ Read carefully — incorrect submissions will **not** be graded.

1. Complete your implementation in `rag_pipeline.py`, `llm_monitor.py`, and `mcp_handler.py`.
2. Make sure the application runs without errors from a clean virtual environment.
3. **Do not** include your `data/` folder, `.env` file, `__pycache__/`, or the `chroma_db/` directory.
4. Zip your project folder:

```
your_student_id_number.zip
```

**Example:** `19036154.zip`

> Names that do not match this format (e.g. `project_final.zip`, `Ali_Kaya.zip`) will be **rejected automatically** and will not be reviewed.

5. Upload your ZIP to the submission link shared by the instructor on the course portal.

**Deadline: Week 12** (exact date on the course portal)

> Duplicate uploads are rejected. Only your **first** valid upload is accepted.

---

## ❓ FAQ

**Q: Can I add extra Python files?**  
A: Yes. You may add helper modules. Just ensure `main.py` still runs by only importing `rag_pipeline`, `llm_monitor`, and `mcp_handler`.

**Q: Which embedding model should I use?**  
A: `all-MiniLM-L6-v2` (free, local, fast) is a great default. You can use OpenAI embeddings if you have credits.

**Q: Does streaming have to work?**  
A: Yes. The `stream_callback` argument in `query()` must be called with individual tokens so the GUI displays output progressively.

**Q: Can I use Ollama instead of a cloud model?**  
A: Absolutely. Select `ollama/llama3` in the model dropdown and make sure Ollama is running locally.

**Q: What MCP server should I implement?**  
A: Any compliant MCP server works. A minimal example is a filesystem tool that reads a specific directory. Refer to the [MCP documentation](https://modelcontextprotocol.io) for details.

---

## 📜 Academic Integrity

All submitted code must be your own work. The use of AI assistants to *understand* concepts is permitted; submitting AI-generated code verbatim as your final submission is not. You may be asked to explain any part of your submission in person.

---

## 🔗 Useful References

- [LangChain Documentation](https://python.langchain.com)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [PySide6 Documentation](https://doc.qt.io/qtforpython-6/)
- [HuggingFace Sentence Transformers](https://www.sbert.net)

---

<p align="center">
  <em>Yıldız Technical University · Mechatronics Engineering · MKT3454</em>
</p>
