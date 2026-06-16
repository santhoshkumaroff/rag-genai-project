# LangGraph Flow

This file documents the LangGraph execution flow used by this project. It focuses on how a user query travels from the CLI into the supervisor agent, how the supervisor delegates work, and how the final answer is returned.

## High-Level Overview

The project uses a multi-agent RAG pattern:

1. `app.py` prepares the document pipeline.
2. `agents/rag_agent.py` builds a supervisor agent.
3. The supervisor decides whether a query should go to the document expert or math expert.
4. The selected expert calls its own tool.
5. The supervisor receives the expert result and responds to the user.

```text
User
 |
 v
app.py CLI loop
 |
 v
Supervisor Agent
 |
 +-- document question --> document_expert --> search_documents --> RAG chain --> FAISS retriever
 |
 +-- math question ------> math_expert ------> calculator
 |
 v
Final assistant response
```

## Files Involved

| File | Responsibility |
|---|---|
| `app.py` | Starts the application, builds the RAG pipeline, creates the agent system, and handles the CLI loop. |
| `agents/rag_agent.py` | Creates the supervisor, document expert, math expert, and delegation tools. |
| `tools/search_tool.py` | Converts the RAG chain into a callable document-search tool. |
| `tools/calculator_tool.py` | Provides the calculator tool for arithmetic queries. |
| `chains/rag_chain.py` | Builds the retrieval chain used by the document expert. |
| `vectorstore/faiss_store.py` | Creates the FAISS vector store and retriever. |

## Startup Flow

When `python app.py` runs, the application builds all dependencies before accepting user input.

```text
validate_config()
 |
 v
load_document("data/sample_document.txt")
 |
 v
get_embeddings()
 |
 v
semantic_chunk_documents(documents, embeddings)
 |
 v
create_vector_store(chunks, embeddings)
 |
 v
get_retriever(vectorstore)
 |
 v
get_llm()
 |
 v
build_rag_chain(retriever, llm)
 |
 v
build_agent(llm, rag_chain, calculator)
 |
 v
CLI ready for user input
```

## Agent Construction Flow

The function `build_agent()` in `agents/rag_agent.py` creates three agents:

- `document_expert`
- `math_expert`
- supervisor agent

```text
build_agent(llm, rag_chain, calculator_tool)
 |
 +--> create_search_tool(rag_chain)
 |
 +--> create document_expert
 |      tools: [search_documents]
 |
 +--> create math_expert
 |      tools: [calculator]
 |
 +--> create delegation tools
 |      - delegate_to_document_expert
 |      - delegate_to_math_expert
 |
 +--> create MemorySaver checkpointer
 |
 +--> create supervisor agent
        tools:
        - delegate_to_document_expert
        - delegate_to_math_expert
```

## Supervisor Flow

The supervisor is the main LangGraph-powered runtime returned by `build_agent()`.

```python
supervisor_agent = create_agent(
    model=llm,
    tools=[delegate_to_document_expert, delegate_to_math_expert],
    checkpointer=memory,
    system_prompt=(...)
)
```

The supervisor receives every user message first. It decides whether to call a tool based on the user request.

| User request type | Supervisor action |
|---|---|
| Document question | Calls `delegate_to_document_expert` |
| Summary request | Calls `delegate_to_document_expert` |
| Arithmetic question | Calls `delegate_to_math_expert` |
| Mixed document and math query | May call one or both delegation tools |
| General response after tool call | Summarizes the specialist result |

## Document Expert Flow

The document expert handles document-grounded questions.

```text
Supervisor
 |
 v
delegate_to_document_expert(query)
 |
 v
document_expert.invoke(...)
 |
 v
search_documents(query)
 |
 v
rag_chain.invoke(query)
 |
 v
retriever gets relevant chunks from FAISS
 |
 v
LLM answers using retrieved context
 |
 v
document_expert returns final content
 |
 v
Supervisor summarizes response
```

The document expert is instructed to only use the document search tool.

## Math Expert Flow

The math expert handles calculation-focused questions.

```text
Supervisor
 |
 v
delegate_to_math_expert(query)
 |
 v
math_expert.invoke(...)
 |
 v
calculator(expression)
 |
 v
math_expert returns final content
 |
 v
Supervisor summarizes response
```

The math expert is instructed to only use the calculator tool.

## Message State

The application invokes the agent system with a `messages` state key:

```python
response = agent_system.invoke(
    {
        "messages": [
            ("user", query)
        ]
    },
    config=config,
)
```

LangGraph-compatible agents maintain conversation state as a list of messages. The final assistant output is extracted from the last message:

```python
final_message = response["messages"][-1]
content = final_message.content
```

## Thread Configuration

`app.py` passes a `thread_id` through the LangGraph config:

```python
thread_id = "production_user_session_1"
config = {"configurable": {"thread_id": thread_id}}
```

The `thread_id` separates checkpointed conversation state. Different thread IDs represent different sessions.

```text
thread_id: production_user_session_1
 |
 v
MemorySaver checkpoint state for that session
```

## Checkpoint Memory

The supervisor uses `MemorySaver`:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
```

This gives the supervisor a checkpointing layer for message state. In this project, the checkpoint memory is in-memory, so it lasts only for the current Python process.

## Example Flow: Document Question

Prompt:

```text
What is the document about?
```

Flow:

```text
User asks document question
 |
 v
Supervisor selects delegate_to_document_expert
 |
 v
document_expert calls search_documents
 |
 v
RAG chain retrieves relevant chunks
 |
 v
LLM creates document-grounded answer
 |
 v
Supervisor returns final answer
```

## Example Flow: Math Question

Prompt:

```text
Calculate 1250 * 0.18
```

Flow:

```text
User asks math question
 |
 v
Supervisor selects delegate_to_math_expert
 |
 v
math_expert calls calculator
 |
 v
Calculator returns result
 |
 v
Supervisor returns final answer
```

## Example Flow: Mixed Query

Prompt:

```text
Summarize the document and calculate 45 + 17.
```

Possible flow:

```text
User asks mixed query
 |
 v
Supervisor delegates summary to document_expert
 |
 v
document_expert uses search_documents and RAG chain
 |
 v
Supervisor delegates arithmetic to math_expert
 |
 v
math_expert uses calculator
 |
 v
Supervisor combines both results
 |
 v
Final response returned to user
```

## Enhanced Flow Variant

The repository also includes `app1.py` and `agents/rag_agent1.py`. These files appear to be an enhanced version of the same flow.

Extra behavior in `app1.py`:

- Supports a `clear` command.
- Creates a new `thread_id` when memory is cleared.
- Adds `recursion_limit: 15` to the config.
- Handles recursion-limit failures with a clearer message.

Extra behavior in `agents/rag_agent1.py`:

- Imports `HumanInTheLoopMiddleware`.
- Applies human approval middleware to delegation tools.
- Uses a more defensive supervisor prompt.

Enhanced config shape:

```python
config = {
    "configurable": {"thread_id": thread_id},
    "recursion_limit": 15
}
```

## Current Limitations

- `MemorySaver` is in-memory only and does not persist after the app exits.
- `requirements.txt` does not explicitly list `langgraph`.
- The calculator currently evaluates expressions directly.
- The active entry point is `app.py`; enhanced behavior in `app1.py` is not wired into the main file.

## Suggested Improvements

- Add `langgraph` to `requirements.txt`.
- Replace direct calculator evaluation with a safer expression parser.
- Move the `clear` and `recursion_limit` behavior from `app1.py` into `app.py`.
- Add tests for routing decisions and session isolation.
- Consider persistent checkpoint storage for production use.
