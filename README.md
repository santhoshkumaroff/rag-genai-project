# RAG GenAI Project

Production-style Retrieval Augmented Generation system built with LangChain, FAISS, and Google / OpenAI LLMs.

For the multi-agent LangGraph supervisor, worker agents, delegation tools, and checkpoint memory flow, see `LANGGRAPH_README.md`.

---

## Architecture

```
DocumentLoader
     │  (PDF / TXT)
     ▼
SemanticChunker          ← topic-aware splitting (no fixed chunk size)
     │
     ▼
EmbeddingFactory         ← Google Generative AI  OR  OpenAI
     │
     ▼
VectorStoreManager       ← FAISS index + similarity retriever
     │
     ├──▶  RAGChain      ← stuff-documents chain + system prompt (anti-hallucination)
     │         │
     │         └──▶  MemoryManager  (InMemoryChatMessageHistory, sliding window)
     │
     └──▶  AgentExecutor ← tool-calling agent
               ├── search_documents  (RAG retrieval)
               └── calculator        (safe arithmetic)
```

---

## Code flow (step-by-step)

This section walks through the runtime flow used by the CLI entry (`app.py`) with concrete examples of the calls and where they're implemented.

1) Validate configuration

```
from config_validator import validate_config
validate_config()
```

Implemented in `config_validator.py`. Ensures required environment variables and keys are present before proceeding.

2) Load documents

```
from loaders.document_loader import load_document
documents = load_document("data/sample_document.txt")
```

`load_document` (in `loaders/document_loader.py`) reads the source file(s) and returns a list of document objects used downstream.

3) Prepare embeddings and chunk documents

```
from embeddings.embedding_factory import get_embeddings
from chunkers.semantic_chunker import semantic_chunk_documents

embeddings = get_embeddings()
chunks = semantic_chunk_documents(documents, embeddings)
```

`get_embeddings()` returns a provider-backed embedding client; `semantic_chunk_documents` (in `chunkers/semantic_chunker.py`) performs semantic splitting and returns chunks ready for indexing.

4) Create vector store and retriever

```
from vectorstore.faiss_store import create_vector_store, get_retriever

vectorstore = create_vector_store(chunks, embeddings)
retriever = get_retriever(vectorstore)
```

This builds the FAISS index and exposes a `retriever` used by the RAG chain.

5) Initialize the LLM

```
from embeddings.llm_factory import get_llm

llm = get_llm()
```

`get_llm()` returns the language model instance (OpenAI/Google) used by both the chain and the agent.

6) Build the RAG chain

```
from chains.rag_chain import build_rag_chain

rag_chain = build_rag_chain(retriever, llm)
```

The RAG chain composes retrieval + prompt templates + the LLM to answer questions strictly from retrieved context.

7) Create tools and the agent

```
from tools.search_tool import create_search_tool
from tools.calculator_tool import calculator
from agents.rag_agent import build_agent

search_tool = create_search_tool(rag_chain)
tools = [search_tool, calculator]
agent = build_agent(llm, tools)
```

`create_search_tool` wraps the RAG chain as a tool; `calculator` is a safe arithmetic tool. `build_agent` wires the LLM + tools into a tool-calling agent.

8) Wrap agent with memory and run interactive loop

```
from langchain_core.runnables.history import RunnableWithMessageHistory
from memory.memory_manager import get_session_history

agent_with_memory = RunnableWithMessageHistory(agent, get_session_history, input_messages_key="messages")

response = agent_with_memory.invoke({"messages": [("user", "What is the summary?")]}, config={"configurable": {"session_id": "user_1"}})
```

`RunnableWithMessageHistory` integrates per-session memory (see `memory/memory_manager.py`) so conversations persist across turns.

9) Text extraction and output

```
def extract_text(response):
    content = response["messages"][-1].content
    if isinstance(content, list):
        return content[0].get("text", "")
    return content

print("Assistant:", extract_text(response))
```

The helper `extract_text` (used in `app.py`) normalises the runnable response into plain text for display.

The above steps are executed in order in `app.py` (see the `main()` function). Use these snippets as a reference when invoking the functionality programmatically or reading through the implementation files.


## Project structure

```
rag-genai-project/
├── app.py
├── config.py
├── config_validator.py
├── requirements.txt
├── .env.example
├── README.md
├── SETUP_GUIDE.md
├── SETUP_GUIDE.txt
├── agents/
│   ├── __init__.py
│   └── rag_agent.py
├── chains/
│   ├── __init__.py
│   └── rag_chain.py
├── chunkers/
│   ├── __init__.py
│   └── semantic_chunker.py
├── data/
│   └── sample_document.txt
├── embeddings/
│   ├── __init__.py
│   ├── embedding_factory.py
│   └── llm_factory.py
├── loaders/
│   ├── __init__.py
│   └── document_loader.py
├── memory/
│   ├── __init__.py
│   └── memory_manager.py
├── prompts/
│   ├── __init__.py
│   ├── agent_prompt.py
│   └── rag_prompt.py
├── tools/
│   ├── __init__.py
│   ├── calculator_tool.py
│   └── search_tool.py
└── vectorstore/
     ├── __init__.py
     └── faiss_store.py
```

---

## Quick start

```bash
# 1. Clone / copy the project
cd rag-genai-project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — add your GOOGLE_API_KEY or OPENAI_API_KEY

# 5. Add a document
# Drop a PDF into data/  (default name: sample_document.pdf)
# OR use the load command at runtime

# 6. Run
python app.py
```

---

## CLI commands

| Command | Description |
|---|---|
| `ask <question>` | Answer via RAG chain with memory |
| `agent <question>` | Answer via tool-calling agent with memory |
| `load <path>` | Load / replace the active document |
| `clear` | Clear conversation memory for current session |
| `quit` | Exit |

Bare input (no prefix) is routed to the agent automatically.

---

## Using as a library

```python
from app import RAGApplication

app = RAGApplication()
app.load_document("data/my_report.pdf")

# RAG chain (answers strictly from context)
answer = app.ask("What were the key findings?")
print(answer)

# Agent (picks the right tool automatically)
answer = app.ask_agent("Summarise section 3 and calculate 1024 * 0.07")
print(answer)

# Multi-turn with explicit session IDs
answer = app.ask("Who wrote this?", session_id="user-42")
answer = app.ask("What else did they publish?", session_id="user-42")

# Clear memory when done
app.clear_memory("user-42")
```

---

## Switching providers

Edit `.env`:

```
# Use OpenAI for both layers
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Or mix: Google embeddings + OpenAI LLM
EMBEDDING_PROVIDER=google
LLM_PROVIDER=openai
```

---

## Key design decisions

**Semantic chunking** — `SemanticChunker` groups sentences by embedding similarity instead of splitting on fixed token counts. This keeps semantically related content together, improving retrieval precision.

**Anti-hallucination system prompt** — The RAG chain's system prompt explicitly forbids the model from using pre-training knowledge that isn't present in the retrieved context. If the context lacks the answer, the model says so.

**InMemoryChatMessageHistory with sliding window** — Each session maintains a rolling window of the last N message pairs, bounding context size while preserving conversational coherence.

**Tool-calling agent** — Rather than always running RAG, the agent decides whether a question needs document retrieval (`search_documents`), arithmetic (`calculator`), or neither. This avoids unnecessary API calls and lets the system handle mixed queries naturally.

**Provider abstraction** — `EmbeddingFactory` and `LLMFactory` hide provider-specific imports behind a single `.build()` call, making it trivial to swap Google ↔ OpenAI via environment variables.
