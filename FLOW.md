# Project Flow

Complete execution flow of the Multi-Agent LangGraph RAG System.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          app.py (async main)                        │
│                                                                     │
│  ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌──────────────┐  │
│  │  Loader   │──▶│ Normalizer │──▶│ Chunker  │──▶│ Vector Store │  │
│  └──────────┘   └────────────┘   └──────────┘   └──────────────┘  │
│                                                              │      │
│                                                              ▼      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     Supervisor Agent                         │   │
│  │  ┌─────────────────────┐      ┌─────────────────────────┐   │   │
│  │  │  document_expert    │      │      math_expert         │   │   │
│  │  │  ┌───────────────┐  │      │  ┌───────────────────┐   │   │   │
│  │  │  │ search_documents│  │      │  │   calculator      │   │   │   │
│  │  │  └───────┬───────┘  │      │  └───────────────────┘   │   │   │
│  │  │          ▼          │      │                           │   │   │
│  │  │     RAG Chain       │      │                           │   │   │
│  │  │  (LCEL pipeline)    │      │                           │   │   │
│  │  └─────────────────────┘      └─────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                      RAGQueryOutput                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Initialization (Startup)

```
python app.py
       │
       ▼
┌──────────────────────┐
│   validate_config()  │  Check GOOGLE_API_KEY exists
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   load_document()    │  Load PDF/TXT from data/
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ normalize_documents()│  Lowercase, unicode, whitespace cleanup
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ get_embeddings()     │  GoogleGenerativeAIEmbeddings
└──────────┬───────────┘
           ▼
┌──────────────────────────────┐
│ semantic_chunk_documents()   │  Split at topic boundaries
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│ create_vector_store()        │  FAISS index from chunks
└──────────┬───────────────────┘
           ▼
┌──────────────────────┐
│   get_retriever()    │  Retriever with top_k=4
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│     get_llm()        │  ChatGoogleGenerativeAI
└──────────┬───────────┘
           ▼
┌──────────────────────────────────────────────────┐
│           build_rag_chain(retriever, llm)         │
│                                                  │
│  LCEL Pipeline:                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────┐    ┌───┐ │
│  │Retriever│───▶│Prompt   │───▶│ LLM │───▶│Out│ │
│  │  + fmt  │    │Template │    │     │    │Pars│ │
│  └─────────┘    └─────────┘    └─────┘    └───┘ │
└──────────────────────────┬───────────────────────┘
                           ▼
┌──────────────────────────────────────────────────┐
│         build_agent(llm, rag_chain, calculator)   │
│                                                  │
│  1. Create document_expert (create_agent)         │
│  2. Create math_expert (create_agent)             │
│  3. Create delegation tools                       │
│  4. Create supervisor with MemorySaver            │
└──────────────────────────┬───────────────────────┘
                           ▼
                    CLI Ready
```

---

## Phase 2: Query Processing (Runtime)

```
User Input: "What was the Q3 revenue?"
       │
       ▼
┌──────────────────────────┐
│  RAGQueryInput (Pydantic)│  Validate query, session_id, top_k
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  agent_system.ainvoke()  │  Async execution
│  {"messages": [user msg]}│
└──────────┬───────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│                SUPERVISOR AGENT                       │
│                                                      │
│  Receives: "What was the Q3 revenue?"                │
│  Decision: Route to document_expert                  │
│                                                      │
│  Calls: delegate_to_document_expert("What was the    │
│          Q3 revenue?")                               │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│              DOCUMENT EXPERT                          │
│                                                      │
│  Calls: search_documents("What was the Q3 revenue?") │
│          │                                           │
│          ▼ (async)                                   │
│  ┌──────────────────────────────────────────────┐    │
│  │            RAG CHAIN (LCEL)                  │    │
│  │                                              │    │
│  │  1. Retriever ──▶ Gets top 4 chunks from FAISS│   │
│  │  2. format_docs() ──▶ Joins chunk texts      │    │
│  │  3. PromptTemplate ──▶ Fills context+question│    │
│  │  4. LLM ──▶ Generates answer                 │    │
│  │  5. StrOutputParser ──▶ Returns string       │    │
│  └──────────────────────────────────────────────┘    │
│          │                                           │
│          ▼                                           │
│  Returns: "Revenue was $50,000 with $12,000 overhead" │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│                SUPERVISOR AGENT                       │
│                                                      │
│  Receives document_expert result                     │
│  Summarizes and formats final response               │
│                                                      │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────┐
│  extract_text(response)  │  Extract last message content
└──────────┬───────────────┘
           ▼
    "Assistant: Revenue was $50,000..."
```

---

## Phase 3: Async Tool Execution

```
┌─────────────────────────────────────────────────────────────┐
│                    ASYNC REALTIME FLOW                       │
│                                                             │
│  User Query                                                 │
│      │                                                      │
│      ▼                                                      │
│  agent_system.ainvoke()  ◄─── asyncio.run()                 │
│      │                                                      │
│      ├──▶ search_documents.ainvoke()                        │
│      │         │                                            │
│      │         ▼                                            │
│      │    rag_chain.ainvoke()                               │
│      │         │                                            │
│      │         ├──▶ retriever.ainvoke()                     │
│      │         ├──▶ prompt.ainvoke()                        │
│      │         ├──▶ llm.ainvoke()                           │
│      │         └──▶ parser.ainvoke()                        │
│      │                                                      │
│      └──▶ calculator.ainvoke()                              │
│                │                                            │
│                ▼                                            │
│           ast.parse() + _safe_eval()                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Models (Pydantic)

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT                                 │
│                                                         │
│  RAGQueryInput:                                         │
│    query: str          "What is RAG?"                   │
│    session_id: str     "production_user_session_1"      │
│    top_k: int          4                                │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    OUTPUT                                │
│                                                         │
│  RAGQueryOutput:                                        │
│    answer: str         "RAG is Retrieval-Augmented..."  │
│    sources: list[str]  ["data/sample_document.txt"]     │
│    confidence: float   0.85                             │
│                                                         │
│  ToolResponse:                                          │
│    tool_name: str      "search_documents"               │
│    result: str         "Found 3 relevant passages..."   │
│    status: str         "success"                        │
└─────────────────────────────────────────────────────────┘
```

---

## Text Normalization Pipeline

```
Raw Document
     │
     ▼
┌────────────────────────┐
│ 1. Unicode normalize   │  NFKD form (accent removal)
│    unicodedata.normalize│
└────────────┬───────────┘
             ▼
┌────────────────────────┐
│ 2. Lowercase           │  text.lower()
└────────────┬───────────┘
             ▼
┌────────────────────────┐
│ 3. Remove special chars│  Regex: keep \w\s.,;:!?
└────────────┬───────────┘
             ▼
┌────────────────────────┐
│ 4. Collapse whitespace │  Regex: \s+ → " "
└────────────┬───────────┘
             ▼
┌────────────────────────┐
│ 5. Limit newlines      │  \n{3,} → \n\n
└────────────┬───────────┘
             ▼
     Normalized Document
```

---

## File Responsibility Map

```
app.py                      Main entry, async loop, Pydantic validation
config.py                   Settings (Pydantic BaseSettings)
config_validator.py         API key check

loaders/document_loader.py  Load PDF/TXT + apply normalization
normalizer/text_normalizer.py  Unicode, lowercase, whitespace cleanup
chunkers/semantic_chunker.py   Topic-boundary splitting

embeddings/embedding_factory.py  Google embeddings
embeddings/llm_factory.py       ChatGoogleGenerativeAI

vectorstore/faiss_store.py   FAISS index + retriever
chains/rag_chain.py          LCEL pipeline (retriever|prompt|llm|parser)

tools/search_tool.py         Async RAG chain wrapper
tools/calculator_tool.py     Async safe math evaluator

agents/rag_agent.py          Supervisor + 2 worker agents (create_agent)
agents/rag_agent1.py         Enhanced variant

models/schemas.py            RAGQueryInput, RAGQueryOutput, ToolResponse, AgentState
prompts/rag_prompt.py        SYSTEM_PROMPT + RAG_PROMPT template
prompts/agent_prompt.py      Agent system prompt (unused)

memory/memory_manager.py     InMemoryChatMessageHistory (unused)
```

---

## Example: Complete Query Flow

**User asks:** "Calculate 1250 * 0.18 and summarize the document"

```
1. RAGQueryInput validates input
2. ainvoke() sends to supervisor
3. Supervisor decides: needs both experts
4. Calls delegate_to_document_expert("summarize the document")
   └──▶ document_expert
        └──▶ search_documents.ainvoke()
             └──▶ rag_chain.ainvoke()
                  ├──▶ FAISS retriever gets chunks
                  ├──▶ format_docs() joins text
                  ├──▶ Prompt fills context + question
                  ├──▶ LLM generates summary
                  └──▶ StrOutputParser returns string
        └──▶ Returns summary to supervisor
5. Calls delegate_to_math_expert("Calculate 1250 * 0.18")
   └──▶ math_expert
        └──▶ calculator.ainvoke("1250 * 0.18")
             ├──▶ ast.parse()
             └──▶ _safe_eval() returns 225.0
        └──▶ Returns "225.0" to supervisor
6. Supervisor combines both results
7. extract_text() pulls final message
8. "Assistant: The document discusses... The calculation result is 225.0"
```
