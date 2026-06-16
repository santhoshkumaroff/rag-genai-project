# LangGraph Functionality

This document explains the LangGraph-based multi-agent functionality in this RAG project. The core implementation lives in `agents/rag_agent.py` and is launched from `app.py`.

## Purpose

The LangGraph layer turns the base RAG pipeline into a supervisor-driven multi-agent system. Instead of sending every user question directly to the document retriever, the supervisor decides which specialist should handle the request:

- `document_expert` answers document and context questions through the RAG search tool.
- `math_expert` answers arithmetic and calculation questions through the calculator tool.
- The supervisor coordinates both specialists and returns the final user-facing answer.

## Runtime Flow

```text
User input
   |
   v
app.py
   |
   v
build_agent(llm, rag_chain, calculator)
   |
   v
Supervisor Agent
   |
   +--> delegate_to_document_expert(query)
   |       |
   |       v
   |   document_expert -> search_documents -> rag_chain -> retriever -> FAISS
   |
   +--> delegate_to_math_expert(query)
           |
           v
       math_expert -> calculator
```

The supervisor and workers are created with LangChain's `create_agent`, which uses LangGraph runtime behavior under the hood. The supervisor is configured with a LangGraph `MemorySaver` checkpointer so conversation state can be tracked per session.

## Main Files

| File | Role |
|---|---|
| `app.py` | CLI entry point that prepares documents, embeddings, vector store, retriever, RAG chain, and agent system. |
| `agents/rag_agent.py` | Builds the active LangGraph-style multi-agent supervisor and specialist agents. |
| `tools/search_tool.py` | Wraps the RAG chain as the `search_documents` tool used by `document_expert`. |
| `tools/calculator_tool.py` | Defines the `calculator` tool used by `math_expert`. |
| `chains/rag_chain.py` | Builds the retrieval-augmented generation chain used by the document specialist. |
| `app1.py` | Enhanced CLI variant with clear-session behavior and recursion-limit configuration. |
| `agents/rag_agent1.py` | Enhanced agent variant with human-in-the-loop middleware comments and setup. |

## Agent Responsibilities

### Supervisor Agent

Defined in `agents/rag_agent.py`:

```python
supervisor_agent = create_agent(
    model=llm,
    tools=[delegate_to_document_expert, delegate_to_math_expert],
    checkpointer=memory,
    system_prompt=(...)
)
```

The supervisor does not directly run document retrieval or calculations. It receives the user message, chooses a delegation tool, waits for the selected specialist to return a result, and summarizes the answer.

### Document Expert

```python
doc_agent = create_agent(
    model=llm,
    tools=[search_tool],
    name="document_expert",
    system_prompt="..."
)
```

The document expert is restricted to the `search_documents` tool. That tool invokes the RAG chain, which retrieves relevant context from the FAISS vector store and generates an answer from the retrieved document chunks.

Use cases:

- Summaries of the uploaded document
- Questions about document facts
- Requests for retrieved context
- Topic or section explanations from the indexed file

### Math Expert

```python
math_agent = create_agent(
    model=llm,
    tools=[calculator_tool],
    name="math_expert",
    system_prompt="..."
)
```

The math expert is restricted to the calculator tool.

Use cases:

- Arithmetic
- Equations
- Numeric expressions
- Mixed questions where part of the answer needs calculation

## Delegation Tools

The supervisor talks to worker agents through two tool wrappers:

```python
@tool
def delegate_to_document_expert(query: str) -> str:
    response = doc_agent.invoke({"messages": [("user", query)]})
    return response["messages"][-1].content

@tool
def delegate_to_math_expert(query: str) -> str:
    response = math_agent.invoke({"messages": [("user", query)]})
    return response["messages"][-1].content
```

These tools bridge supervisor state into specialist-agent calls. Each specialist receives a normal user message and returns its final message content to the supervisor.

## Memory and Sessions

The active agent uses LangGraph checkpoint memory:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
```

In `app.py`, each invocation passes a `thread_id`:

```python
config = {"configurable": {"thread_id": "production_user_session_1"}}
```

LangGraph uses this configurable thread ID to separate checkpointed state across conversations. If two conversations use different `thread_id` values, they do not share the same supervisor memory.

## Running the LangGraph Agent

Install dependencies and configure environment variables as described in `README.md` or `SETUP_GUIDE.md`, then run:

```bash
python app.py
```

Example prompts:

```text
User: Summarize the document.
User: What are the main points in the uploaded file?
User: Calculate 1250 * 0.18
User: Summarize the document and calculate 45 + 17
```

Type `exit` to stop the CLI session.

## Enhanced Variant

The repo also contains `app1.py` and `agents/rag_agent1.py`, which appear to be an enhanced variant of the current LangGraph flow.

Additional behavior in `app1.py`:

- `clear` command creates a fresh `thread_id`.
- `recursion_limit` is set to `15` to reduce runaway loops.
- Recursion-limit errors are caught and reported as safety stops.

Additional behavior in `agents/rag_agent1.py`:

- Imports `HumanInTheLoopMiddleware`.
- Configures middleware for supervisor delegation tools.
- Uses a more defensive supervisor prompt that asks the system to report specialist failures cleanly.

To use that variant directly:

```bash
python app1.py
```

## Configuration Notes

The code imports `langgraph.checkpoint.memory.MemorySaver`, so the environment must include LangGraph-compatible LangChain packages. If dependency installation fails with `ModuleNotFoundError: No module named 'langgraph'`, add LangGraph explicitly:

```bash
pip install langgraph
```

The existing `requirements.txt` includes the main LangChain packages, but it does not currently list `langgraph` as a direct dependency.

## Extension Ideas

- Add more specialists, such as a web-search expert, SQL expert, or summarization expert.
- Replace in-memory checkpointing with persistent checkpoint storage for long-running sessions.
- Add stricter tool safety around calculator execution.
- Promote the enhanced `app1.py` and `rag_agent1.py` flow into the main `app.py` path after testing.
- Add tests for supervisor routing, document delegation, math delegation, and session isolation.
