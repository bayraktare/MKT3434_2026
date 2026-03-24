# 🤖 Multi-Agent RAG + MCP System
### MKT3434 — Introduction to Machine Learning · Term Project
**Yıldız Technical University · Mechatronics Engineering Department**

---

> An advanced retrieval-augmented generation system in which multiple specialised LLM agents collaborate, critique, and refine responses over heterogeneous document collections — orchestrated through a dynamic control layer and extended via the Model Context Protocol.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Core Agents](#-core-agents)
- [Advanced Agents](#-advanced-agents)
- [Advanced Features](#-advanced-features)
- [Default Rules System](#-default-rules--guidelines)
- [Use Case Scenarios](#-use-case-scenarios)
- [Monitoring & Evaluation](#-monitoring--evaluation)
- [Ablation Study](#-ablation-study)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Design Decisions](#-design-decisions)
- [Submission](#-submission)

---

## 📌 Overview

Standard RAG pipelines pass a user query directly to a retriever and then to a generator, producing a single unverified response. This project replaces that flat architecture with a **pipeline of specialised agents**, each responsible for a distinct cognitive task.

The system is capable of:

- Decomposing complex, multi-part queries before retrieval
- Routing queries to domain-relevant document subsets
- Retrieving and re-ranking evidence from 500+ page, multi-format corpora
- Synthesizing grounded answers from retrieved chunks
- Critically evaluating every answer for faithfulness and completeness
- Deciding when to delegate to external tools via MCP
- Applying user-defined rules that constrain writing style, citation format, and output structure
- Logging every agent interaction for session-level analysis and ablation

---

## 🏗 Architecture

```
                     ┌──────────────────────────────────┐
                     │     USER  QUERY  +  RULES         │
                     └─────────────────┬────────────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────┐
                     │       ORCHESTRATOR  AGENT        │
                     │  • Parses query and active rules │
                     │  • Plans agent execution order   │
                     │  • Handles retries on low score  │
                     │  • Streams final answer to GUI   │
                     └──┬──────┬─────────┬──────┬───────┘
                        │      │         │      │
            ┌───────────┘      │         │      └─────────────┐
            ▼                  ▼         ▼                    ▼
┌───────────────────┐  ┌───────────┐  ┌──────────────┐  ┌─────────────┐
│  QUERY            │  │  DOMAIN   │  │  TOOL / MCP  │  │  MEMORY /   │
│  REFORMULATOR     │  │  ROUTER   │  │  AGENT       │  │  REFLECTION │
│                   │  │           │  │              │  │  (optional) │
│ • Expand & clarify│  │ • Classify│  │ • Calls MCP  │  │ • Recalls   │
│ • Decompose into  │  │   domain  │  │   tools when │  │   prior     │
│   sub-queries     │  │ • Select  │  │   needed     │  │   turns     │
│ • Identify intent │  │   doc     │  │ • Returns    │  │ • Reflects  │
│                   │  │   subset  │  │   tool output│  │   on errors │
└────────┬──────────┘  └─────┬─────┘  └──────┬───────┘  └─────────────┘
         │                   │               │
         └──────────┬────────┘               │
                    ▼                        │
      ┌─────────────────────────┐            │
      │      RETRIEVER  AGENT   │◄───────────┘
      │  • Hybrid search        │
      │    (dense + sparse)     │
      │  • Cross-encoder        │
      │    re-ranking           │
      │  • Returns chunks +     │
      │    provenance metadata  │
      └─────────────┬───────────┘
                    │
                    ▼
      ┌─────────────────────────┐
      │    SYNTHESIZER  AGENT   │
      │  • Applies active rules │
      │  • Generates grounded   │
      │    answer from chunks   │
      │  • Merges sub-answers   │
      └─────────────┬───────────┘
                    │
                    ▼
      ┌─────────────────────────┐
      │      CRITIC  AGENT      │
      │  • Scores faithfulness  │
      │  • Scores completeness  │
      │  • Flags hallucinations │
      │  • Returns JSON report  │
      └─────────────┬───────────┘
                    │
           score ≥ threshold?
            ┌───────┴───────┐
           YES              NO
            │               └──► Orchestrator retries
            ▼                    with revised strategy
     Final Answer +
     Quality Report → GUI
```

### Data Flow Summary

1. The **Orchestrator** receives the query alongside any active rules, then plans execution.
2. The **Query Reformulator** rewrites and decomposes the query; results feed the **Domain Router** and **Retriever**.
3. The **Domain Router** narrows the search space to the relevant document subset.
4. The **Retriever** performs hybrid search and re-ranks results; the **Tool/MCP Agent** handles any external tool calls in parallel.
5. The **Synthesizer** generates a response constrained by active rules.
6. The **Critic** evaluates the response and returns a structured quality report.
7. If the score falls below threshold, the Orchestrator adjusts strategy and retries; otherwise it streams the final answer.

---

## 🧠 Core Agents

### Orchestrator Agent
The central coordinator. It receives the raw query, loads active rules, determines which agents to call and in what order, manages the retry loop based on the Critic score, and streams the final response. All other agents are invoked only through the Orchestrator.

### Query Reformulator Agent
Analyses the raw query and produces improved versions: expands abbreviations, resolves ambiguities, identifies the domain, and decomposes multi-part questions into focused sub-queries that can be answered independently and then merged.

### Retriever Agent
Takes reformulated queries and retrieves the most relevant chunks using **hybrid search** (dense vector similarity + optional BM25 sparse retrieval). Applies a cross-encoder **re-ranker** to reorder results and returns chunks annotated with source file, page, and similarity score.

### Synthesizer Agent
Receives ranked chunks and generates a grounded, coherent answer. Applies all active rules — citation format, writing style, length constraints — as explicit instructions in its system prompt. For multi-part queries it merges partial answers from multiple Retriever calls into a unified response.

### Critic Agent
Evaluates the Synthesizer's output against source chunks. Returns a structured JSON report containing faithfulness, completeness, hallucination flags, and an overall quality score. This score drives the Orchestrator's retry decision and is displayed in the Monitor tab.

---

## ⚙️ Advanced Agents

### Domain Router Agent
Classifies the query into one of the configured domain categories (e.g. health, engineering, sports, politics) and selects the corresponding document subset from the vector store. This prevents irrelevant chunks from polluting retrieval and significantly improves precision on heterogeneous corpora.

### Tool / MCP Agent
Decides at runtime whether the current query requires a tool call — numerical calculation, structured data lookup, or direct file read — rather than (or in addition to) document retrieval. Communicates with an MCP server via `mcp_handler`. Available tools are enumerated with `list_tools()` and invoked with `call_tool(name, args)`. Tool output is passed back as additional context to the Synthesizer.

| Example Tool | Description |
|---|---|
| `calculator` | Evaluates numerical or unit-conversion expressions |
| `file_reader` | Reads a specific file path on demand |
| `structured_lookup` | Queries a local JSON/CSV knowledge base |

### Memory / Reflection Agent *(optional)*
Maintains a short-term session memory of previous queries and answers. At the start of a new query it injects relevant prior context into the Orchestrator's plan. After a failed retry it reflects on what went wrong and suggests a different retrieval or synthesis strategy.

---

## ⚙️ Advanced Features

### 1 · Adaptive Orchestration

The Orchestrator does not follow a fixed script. It evaluates the complexity of each query and:

- **Skips** the Reformulator for simple, unambiguous questions
- **Runs agents in parallel** (e.g. Domain Router + Reformulator) using `asyncio.gather` when outputs are independent
- **Retries** with a revised prompt or expanded chunk set if the Critic score is below threshold
- **Escalates** to the Tool/MCP Agent if retrieval alone is insufficient

### 2 · Improved Retrieval

| Technique | Description |
|---|---|
| Dense retrieval | Cosine similarity over embeddings (`all-MiniLM-L6-v2` or OpenAI) |
| Sparse retrieval | Optional BM25 keyword matching for exact-term queries |
| Cross-encoder re-ranking | `ms-marco-MiniLM-L-6-v2` scores every (query, chunk) pair |
| Query expansion | Reformulator generates synonym-enriched variants before retrieval |
| Domain filtering | Domain Router reduces the search space before similarity search |

### 3 · Prompt Engineering Layer

All agent prompts live in a single `prompts.py` module. Each agent has a dedicated system prompt encoding its role, output format, and any active rules. Prompts are parameterised with f-strings so rules, context, and query are injected at runtime. This centralised design makes ablation and experimentation straightforward.

### 4 · Evaluation Framework

Every agent call is recorded by `LLMMonitor`:

| Field | Description |
|---|---|
| `agent_role` | Which agent made the call |
| `model` | LLM used |
| `prompt_tokens` | Input token count |
| `completion_tokens` | Output token count |
| `latency_sec` | Wall-clock time |
| `faithfulness` | Critic score (0–1) |
| `completeness` | Critic score (0–1) |
| `hallucinations` | List of flagged unsupported claims |

Aggregate statistics are computed per role across the session and displayed in the Monitor tab. The full log is exportable as JSON.

---

## 📏 Default Rules & Guidelines

Students can upload a plain-text or JSON rules file before sending a query. Active rules are parsed by the Orchestrator and injected into the relevant agent prompts.

**Example rules file:**

```json
{
  "citation_style": "IEEE",
  "writing_style": "formal academic",
  "output_structure": ["Introduction", "Method", "Results", "Conclusion"],
  "max_response_length": 500,
  "domain_constraint": "engineering",
  "language": "English"
}
```

**How rules are applied:**

- The **Synthesizer** receives rules as explicit instructions in its system prompt, constraining tone, structure, and citation format.
- The **Critic** uses the same rules as evaluation criteria — if `citation_style` is IEEE, it penalises uncited claims or incorrect format in its faithfulness score.
- The **Domain Router** uses `domain_constraint` to hard-filter the document subset before retrieval.

This mechanism allows the same system to behave very differently for a formal engineering report versus a concise research summary — without changing any code.

---

## 🎓 Use Case Scenarios

### Academic Paper Writing
**Workflow:** Reformulator decomposes the research question → Domain Router restricts to academic sources → Retriever fetches relevant papers → Synthesizer writes in IEEE style per active rules → Critic checks citation accuracy.  
**Example:** *"Summarise the state of the art in simultaneous localisation and mapping using neural radiance fields."*

### Technical Report Generation
**Workflow:** Orchestrator detects a structured-output rule → Synthesizer produces sections (Introduction, Method, Results) → Tool Agent calls a unit-conversion tool for numerical data → Critic evaluates completeness against the defined output structure.  
**Example:** *"Generate a technical comparison of PID and LQR controllers for a 2-DOF robotic arm."*

### Project Proposal Writing
**Workflow:** Memory Agent surfaces previously discussed ideas → Reformulator expands the brief → Synthesizer writes a structured proposal following active rules → Critic ensures objectives are supported by retrieved literature.  
**Example:** *"Write a project proposal for an edge-deployed defect detection system in manufacturing."*

### General Research Assistance
**Workflow:** Domain Router classifies across multiple domains → Retriever performs broad hybrid search → Synthesizer summarises findings concisely → Critic flags any speculative statements not grounded in the source documents.  
**Example:** *"What are the main health risks of fine particulate matter exposure, and what engineering controls mitigate them?"*

---

## 📊 Monitoring & Evaluation

The **Monitor tab** displays a live feed of every agent call in the current session. After each query, the Critic's JSON quality report is appended. At session end, aggregate statistics are shown per agent role.

```
┌──────────────────┬────────┬──────────┬──────────┬────────────────────┐
│  Agent Role      │ Calls  │  Avg ms  │  Tokens  │  Avg Faithfulness  │
├──────────────────┼────────┼──────────┼──────────┼────────────────────┤
│  Reformulator    │  5     │  820     │  1 240   │  —                 │
│  Domain Router   │  5     │  310     │  480     │  —                 │
│  Retriever       │  8     │  1 540   │  —       │  —                 │
│  Synthesizer     │  6     │  2 100   │  9 830   │  —                 │
│  Critic          │  6     │  1 200   │  4 210   │  0.87              │
│  Tool / MCP      │  2     │  230     │  340     │  —                 │
└──────────────────┴────────┴──────────┴──────────┴────────────────────┘
```

---

## 🧪 Ablation Study

Your report must include an ablation study that isolates the contribution of each major component. Run the same set of at least five queries under each condition and compare Critic faithfulness and completeness scores.

| Configuration | What is removed |
|---|---|
| **Full system** | Baseline — all agents active |
| **Without Reformulator** | Raw query sent directly to Retriever |
| **Without Critic** | Synthesizer output accepted without evaluation; no retries |
| **Without Re-ranking** | Retriever returns top-K by embedding similarity only |
| **Without Domain Router** | Full corpus searched for every query |
| **Without Rules** | Synthesizer receives no style or structure constraints |

Present results as a table of average scores per configuration and discuss where the system degrades most and why.

---

## 📁 Project Structure

```
MKT3434_project/
│
├── main.py              ← PySide6 GUI shell          
├── orchestrator.py      ← Orchestrator Agent         
├── agents.py            ← All agent classes           
├── rag_pipeline.py      ← Vector store & indexing   
├── mcp_handler.py       ← MCP server connection      
├── llm_monitor.py       ← Logging & evaluation      
├── prompts.py           ← Centralised prompt library 
│
├── requirements.txt
├── .gitignore
├── README.md
│
└── data/               ← Your documents (git-ignored)
    ├── health/
    ├── engineering/
    ├── sports/
    └── politics/
```

> `main.py` imports only `orchestrator`, `rag_pipeline`, `mcp_handler`, and `llm_monitor`. Do **not** rename their public methods. You may add as many helper modules as needed.

---

## 🚀 Setup

### 1 — Clone

```bash
git clone https://github.com/<instructor-handle>/MKT3434-term-project.git
cd MKT3434-term-project
```

### 2 — Virtual environment

```bash
python -m venv .venv
# Windows:       .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> On Windows, `faiss-cpu` may require [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### 4 — API keys

```env
# .env  (git-ignored)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...        # optional — Gemini
```

Or paste into the GUI's **API KEY** field at runtime.

### 5 — Run

```bash
python main.py
```

---

## 💡 Design Decisions

**Why multi-agent instead of single-agent?**  
A single LLM call conflates retrieval quality, answer generation, and evaluation into one opaque step. Separating concerns into agents makes each step independently auditable, replaceable, and tunable without affecting the rest of the pipeline.

**Why is the Critic essential?**  
Without evaluation there is no feedback mechanism. The Critic closes the loop: a low score triggers a retry with different context or a different strategy, preventing hallucinated or incomplete answers from reaching the user silently.

**Why hybrid retrieval?**  
Dense retrieval excels at semantic similarity but can miss exact-term matches — acronyms, model numbers, proper nouns. Sparse BM25 handles these cases. Combining both produces more robust recall across the heterogeneous document types in this project.

**Why a centralised prompt layer?**  
Scattered hard-coded prompts are impossible to compare systematically. A single `prompts.py` makes prompt ablations trivial, enforces consistency across agents, and allows rule injection to be managed in one place.

---

## 🏁 Conclusion

This system demonstrates that RAG becomes substantially more reliable and controllable when decomposed into a pipeline of specialised agents. The Critic-driven retry loop, domain-aware routing, and hybrid retrieval together address the three most common failure modes of naive RAG: poor recall, hallucination, and lack of domain focus.

The architecture maps directly onto production-grade agentic systems used in industry today. Natural extensions — a persistent memory store, an active learning loop over Critic scores, a richer MCP toolset — would yield a system deployable in real document-heavy workflows such as engineering documentation, legal research, or academic literature review.

---

## 📦 Submission

> ⚠️ Non-compliant submissions are rejected automatically and will not be graded.

- All required agents must be functional and exercised during a query.
- Include `report.pdf` (max 6 pages): architecture decisions, ablation results, sample outputs.
- Exclude: **model weight files.**

**Name your ZIP as your student ID number only:**

```
19036154.zip   ✅       project_final.zip  ❌       Ali_Kaya.zip  ❌
```

Upload to the submission link on the course portal.  
**Deadline: Week 12.** All uploads are logged — only your **first** valid upload counts.

---

## 🔗 References

[LangChain](https://python.langchain.com) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [ChromaDB](https://docs.trychroma.com) · [Model Context Protocol](https://modelcontextprotocol.io) · [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) · [OpenAI SDK](https://github.com/openai/openai-python) · [Sentence Transformers](https://www.sbert.net) · [PySide6](https://doc.qt.io/qtforpython-6/)

---

## 📜 Academic Integrity

All submitted code must be your own work. You may use AI tools to understand concepts and debug; submitting AI-generated code verbatim as your final submission is a violation of academic integrity policy. You may be asked to explain any part of your work in person.

---

<p align="center"><em>Yıldız Technical University · Mechatronics Engineering · MKT3434 · Introduction to Machine Learning</em></p>
