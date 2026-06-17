# Project Flow

Complete execution flow of the Multi-Agent LangGraph RAG System.

---

## Architecture Overview

```
+=====================================================================+
|                         app.py (async main)                          |
|                                                                      |
|  +----------+    +------------+    +----------+    +---------------+  |
|  |  Loader  |--->| Normalizer |--->| Chunker  |--->| Vector Store  |  |
|  +----------+    +------------+    +----------+    +-------+-------+  |
|                                                        |              |
|                                                        v              |
|  +--------------------------------------------------------------+    |
|  |                      Supervisor Agent                         |    |
|  |                                                              |    |
|  |  +------------------+  +--------------+  +----------------+  |    |
|  |  | document_expert  |  |  math_expert |  |   web_expert   |  |    |
|  |  |                  |  |              |  |                |  |    |
|  |  | +--------------+ |  | +----------+ |  | +------------+ |  |    |
|  |  | |search_documents| |  | |calculator| |  | | web_search | |  |    |
|  |  | |web_search    | |  | +----------+ |  | | (DuckDuckGo)| |  |    |
|  |  | +------+-------+ |  |              |  | +------------+ |  |    |
|  |  |        |         |  |              |  |                |  |    |
|  |  |        v         |  |              |  |                |  |    |
|  |  |   RAG Chain      |  |              |  |                |  |    |
|  |  |   (LCEL)         |  |              |  |                |  |    |
|  |  +------------------+  +--------------+  +----------------+  |    |
|  +--------------------------------------------------------------+    |
|                                |                                     |
|                                v                                     |
|                        RAGQueryOutput                                |
+=====================================================================+
```

---

## Phase 1: Initialization (Startup)

```
python app.py
    |
    v
+----------------------------+
|    validate_config()       |  Check GOOGLE_API_KEY exists
+-------------+--------------+
              |
              v
+----------------------------+
|    load_document()         |  Load PDF/TXT from data/
+-------------+--------------+
              |
              v
+----------------------------+
|  normalize_documents()     |  Lowercase, unicode, whitespace cleanup
+-------------+--------------+
              |
              v
+----------------------------+
|    get_embeddings()        |  GoogleGenerativeAIEmbeddings
+-------------+--------------+
              |
              v
+----------------------------+
| semantic_chunk_documents() |  Split at topic boundaries
+-------------+--------------+
              |
              v
+----------------------------+
|  create_vector_store()     |  FAISS index from chunks
+-------------+--------------+
              |
              v
+----------------------------+
|    get_retriever()         |  Retriever with top_k=4
+-------------+--------------+
              |
              v
+----------------------------+
|      get_llm()             |  ChatGoogleGenerativeAI
+-------------+--------------+
              |
              v
+------------------------------------------------------------+
|            build_rag_chain(retriever, llm)                  |
|                                                            |
|   LCEL Pipeline:                                           |
|   +-----------+    +-----------+    +-----+    +--------+  |
|   | Retriever |--->|  Prompt   |--->| LLM |--->| Output |  |
|   |  + fmt    |    | Template  |    |     |    | Parser |  |
|   +-----------+    +-----------+    +-----+    +--------+  |
+---------------------------+--------------------------------+
                            |
                            v
+------------------------------------------------------------+
|          build_agent(llm, rag_chain, calculator)            |
|                                                            |
|   1. Create document_expert (create_agent)                 |
|      tools: [search_documents, web_search]                 |
|   2. Create math_expert (create_agent)                     |
|   3. Create web_expert (create_agent)                      |
|      tools: [web_search]                                   |
|   4. Create delegation tools                               |
|   5. Create supervisor with MemorySaver                    |
+---------------------------+--------------------------------+
                            |
                            v
                      CLI Ready
```

---

## Phase 2: Query Processing (Runtime)

```
User Input: "What was the Q3 revenue?"
    |
    v
+----------------------------+
|  RAGQueryInput (Pydantic)  |  Validate query, session_id, top_k
+-------------+--------------+
              |
              v
+----------------------------+
|  agent_system.ainvoke()   |  Async execution
|  {"messages": [user msg]}  |
+-------------+--------------+
              |
              v
+----------------------------------------------------------+
|                    SUPERVISOR AGENT                       |
|                                                          |
|   Receives: "What was the Q3 revenue?"                   |
|   Decision: Route to document_expert                     |
|                                                          |
|   Calls: delegate_to_document_expert("What was the       |
|           Q3 revenue?")                                  |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
|                   DOCUMENT EXPERT                         |
|                                                          |
|   Calls: search_documents("What was the Q3 revenue?")    |
|       |                                                  |
|       v (async)                                          |
|   +--------------------------------------------------+   |
|   |               RAG CHAIN (LCEL)                   |   |
|   |                                                  |   |
|   |  1. Retriever --> Gets top 4 chunks from FAISS   |   |
|   |  2. format_docs() --> Joins chunk texts           |   |
|   |  3. PromptTemplate --> Fills context + question   |   |
|   |  4. LLM --> Generates answer                      |   |
|   |  5. StrOutputParser --> Returns string            |   |
|   +--------------------------------------------------+   |
|       |                                                  |
|       v                                                  |
|   Returns: "Revenue was $50,000 with $12,000 overhead"    |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
|                    SUPERVISOR AGENT                       |
|                                                          |
|   Receives document_expert result                        |
|   Summarizes and formats final response                  |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------+
|   extract_text(response)   |  Extract last message content
+-------------+--------------+
              |
              v
   "Assistant: Revenue was $50,000..."
```

---

## Phase 3: Async Tool Execution

```
+---------------------------------------------------------------+
|                     ASYNC REALTIME FLOW                        |
|                                                               |
|   User Query                                                  |
|       |                                                       |
|       v                                                       |
|   agent_system.ainvoke()  <--- asyncio.run()                  |
|       |                                                       |
|       +---> search_documents.ainvoke()                        |
|       |         |                                             |
|       |         v                                             |
|       |    rag_chain.ainvoke()                                |
|       |         |                                             |
|       |         +---> retriever.ainvoke()                     |
|       |         +---> prompt.ainvoke()                        |
|       |         +---> llm.ainvoke()                           |
|       |         +---> parser.ainvoke()                        |
|       |                                                       |
|       +---> calculator.ainvoke()                              |
|       |         |                                             |
|       |         v                                             |
|       |    ast.parse() + _safe_eval()                         |
|       |                                                       |
|       +---> web_search.ainvoke()                              |
|                 |                                             |
|                 v                                             |
|            DuckDuckGoSearchResults                           |
|                 |                                             |
|                 v                                             |
|            Format + return results                            |
|                                                               |
+---------------------------------------------------------------+
```

---

## Tool Examples

### 1. search_documents (RAG Chain)

Searches local documents using FAISS vector store + LLM.

```
Input:  "What is the revenue mentioned in the document?"

Execution:
  +-----------------------------------------------------------+
  |  search_documents.ainvoke("What is the revenue...")        |
  |       |                                                   |
  |       v                                                   |
  |  rag_chain.ainvoke(query)                                 |
  |       |                                                   |
  |       +---> retriever.get_relevant_documents(query)        |
  |       |     Returns: [Document(page_content="Revenue       |
  |       |     Q3 2024 was $50,000...", metadata={...})]      |
  |       |                                                   |
  |       +---> format_docs(documents)                         |
  |       |     Returns: "Revenue Q3 2024 was $50,000..."      |
  |       |                                                   |
  |       +---> prompt.format(context=..., question=...)       |
  |       |     Returns: "Context: Revenue Q3...\nQuestion:    |
  |       |     What is the revenue?\nAnswer:"                 |
  |       |                                                   |
  |       +---> llm.ainvoke(prompt)                            |
  |       |     Returns: "The document mentions Q3 revenue     |
  |       |     of $50,000."                                   |
  |       |                                                   |
  |       +---> StrOutputParser().parse(response)              |
  |             Returns: "The document mentions Q3 revenue     |
  |             of $50,000."                                   |
  +-----------------------------------------------------------+

Output: "The document mentions Q3 revenue of $50,000."
```

---

### 2. calculator (Safe Math Evaluator)

Evaluates arithmetic expressions safely using AST parsing.

```
Input:  "1250 * 0.18"

Execution:
  +-----------------------------------------------------------+
  |  calculator.ainvoke("1250 * 0.18")                         |
  |       |                                                   |
  |       v                                                   |
  |  ast.parse("1250 * 0.18", mode="eval")                    |
  |       |                                                   |
  |       |   AST Tree:                                       |
  |       |   Expression(                                     |
  |       |     body=BinOp(                                   |
  |       |       left=Constant(value=1250),                  |
  |       |       op=Mult(),                                  |
  |       |       right=Constant(value=0.18)                  |
  |       |     )                                             |
  |       |   )                                               |
  |       |                                                   |
  |       v                                                   |
  |  _safe_eval(tree)                                         |
  |       |                                                   |
  |       +---> _safe_eval(BinOp)                             |
  |       |     +---> _safe_eval(left=1250) --> 1250          |
  |       |     +---> _safe_eval(right=0.18) --> 0.18         |
  |       |     +---> operator.mul(1250, 0.18) --> 225.0      |
  |       |                                                   |
  |       v                                                   |
  |  ToolResponse(tool_name="calculator", result="225.0",     |
  |               status="success")                           |
  +-----------------------------------------------------------+

Output: "225.0"
```

```
Input:  "100 / 0"  (Division by zero)

Execution:
  +-----------------------------------------------------------+
  |  calculator.ainvoke("100 / 0")                             |
  |       |                                                   |
  |       v                                                   |
  |  ast.parse("100 / 0", mode="eval")                        |
  |       |                                                   |
  |       v                                                   |
  |  _safe_eval(tree)                                         |
  |       |                                                   |
  |       +---> ZeroDivisionError caught                      |
  |       |                                                   |
  |       v                                                   |
  |  ToolResponse(tool_name="calculator",                     |
  |               result="Invalid calculation",               |
  |               status="error")                             |
  +-----------------------------------------------------------+

Output: "Invalid calculation"
```

```
Input:  "__import__('os').system('rm -rf /')"  (Injection attempt)

Execution:
  +-----------------------------------------------------------+
  |  calculator.ainvoke("__import__('os')...")                 |
  |       |                                                   |
  |       v                                                   |
  |  ast.parse("__import__('os')...", mode="eval")            |
  |       |                                                   |
  |       |   AST Tree contains: Import/Call nodes            |
  |       |   NOT in SAFE_OPS dictionary                      |
  |       |                                                   |
  |       v                                                   |
  |  _safe_eval(tree)                                         |
  |       |                                                   |
  |       +---> ValueError: "Unsafe or unsupported expression"|
  |       |                                                   |
  |       v                                                   |
  |  ToolResponse(tool_name="calculator",                     |
  |               result="Invalid calculation",               |
  |               status="error")                             |
  +-----------------------------------------------------------+

Output: "Invalid calculation"  (Blocked safely)
```

---

### 3. web_search (DuckDuckGo Real-Time Search)

Searches the web for real-time information using DuckDuckGo (free, no API key).

```
Input:  "What is the latest news about AI?"

Execution:
  +-----------------------------------------------------------+
  |  web_search.ainvoke("What is the latest news about AI?")   |
  |       |                                                   |
  |       v                                                   |
  |  DuckDuckGoSearchResults(max_results=3).ainvoke(query)     |
  |       |                                                   |
  |       |   Search via DuckDuckGo:                           |
  |       |   Query: "What is the latest news about AI?"       |
  |       |   Max results: 3                                   |
  |       |                                                   |
  |       v                                                   |
  |  Raw Results:                                             |
  |  [                                                        |
  |    {                                                      |
  |      "title": "AI Breakthroughs in 2024",                 |
  |      "content": "Major advances in LLMs...",              |
  |      "url": "https://example.com/article1"                |
  |    },                                                     |
  |    {                                                      |
  |      "title": "New AI Regulations",                       |
  |      "content": "Governments worldwide...",               |
  |      "url": "https://example.com/article2"                |
  |    }                                                      |
  |  ]                                                        |
  |       |                                                   |
  |       v                                                   |
  |  Format:                                                  |
  |  "**AI Breakthroughs in 2024**\nMajor advances..."       |
  |  + "\n\n**New AI Regulations**\nGovernments..."           |
  |       |                                                   |
  |       v                                                   |
  |  ToolResponse(tool_name="web_search",                     |
  |               result=formatted_text,                      |
  |               status="success")                           |
  +-----------------------------------------------------------+

Output: "**AI Breakthroughs in 2024**
         Major advances in LLMs...

         **New AI Regulations**
         Governments worldwide..."
```

```
Input:  "xyz123 nonesense query"  (No results found)

Execution:
  +-----------------------------------------------------------+
  |  web_search.ainvoke("xyz123 nonesense query")              |
  |       |                                                   |
  |       v                                                   |
  |  DuckDuckGoSearchResults(max_results=3).ainvoke(query)     |
  |       |                                                   |
  |       v                                                   |
  |  Raw Results: []  (Empty array)                           |
  |       |                                                   |
  |       v                                                   |
  |  formatted = ""  (Empty string)                           |
  |       |                                                   |
  |       v                                                   |
  |  ToolResponse(tool_name="web_search",                     |
  |               result="No results found",                  |
  |               status="success")                           |
  +-----------------------------------------------------------+

Output: "No results found"
```

---

### 4. delegate_to_document_expert (Supervisor Delegation)

Supervisor delegates document questions to the document expert.

```
Input:  "Summarize the document about RAG"

Execution:
  +-----------------------------------------------------------+
  |  delegate_to_document_expert("Summarize the document...")  |
  |       |                                                   |
  |       v                                                   |
  |  doc_agent.invoke({"messages": [("user", query)]})        |
  |       |                                                   |
  |       |   Document Expert内部:                            |
  |       |   +-------------------------------------------+   |
  |       |   | Agent decides to call search_documents    |   |
  |       |   |           |                               |   |
  |       |   |           v                               |   |
  |       |   | search_documents.ainvoke(query)           |   |
  |       |   |           |                               |   |
  |       |   |           v                               |   |
  |       |   | rag_chain.ainvoke(query)                  |   |
  |       |   |           |                               |   |
  |       |   |           v                               |   |
  |       |   | Returns: "RAG combines retrieval with    |   |
  |       |   | generation to improve accuracy..."       |   |
  |       |   +-------------------------------------------+   |
  |       |                                                   |
  |       v                                                   |
  |  response["messages"][-1].content                         |
  |       |                                                   |
  |       v                                                   |
  |  Returns: "RAG combines retrieval with generation..."     |
  +-----------------------------------------------------------+

Output: "RAG combines retrieval with generation to improve accuracy..."
```

---

### 5. delegate_to_math_expert (Supervisor Delegation)

Supervisor delegates math questions to the math expert.

```
Input:  "Calculate 1250 * 0.18"

Execution:
  +-----------------------------------------------------------+
  |  delegate_to_math_expert("Calculate 1250 * 0.18")         |
  |       |                                                   |
  |       v                                                   |
  |  math_agent.invoke({"messages": [("user", query)]})       |
  |       |                                                   |
  |       |   Math Expert内部:                                |
  |       |   +-------------------------------------------+   |
  |       |   | Agent decides to call calculator         |   |
  |       |   |           |                               |   |
  |       |   |           v                               |   |
  |       |   | calculator.ainvoke("1250 * 0.18")        |   |
  |       |   |           |                               |   |
  |       |   |           v                               |   |
  |       |   | Returns: "225.0"                          |   |
  |       |   +-------------------------------------------+   |
  |       |                                                   |
  |       v                                                   |
  |  response["messages"][-1].content                         |
  |       |                                                   |
  |       v                                                   |
  |  Returns: "225.0"                                         |
  +-----------------------------------------------------------+

Output: "225.0"
```

---

### 6. delegate_to_web_expert (Supervisor Delegation)

Supervisor delegates real-time questions to the web expert.

```
Input:  "What is the current stock price of Apple?"

Execution:
  +-----------------------------------------------------------+
  |  delegate_to_web_expert("What is the current stock        |
  |                          price of Apple?")                |
  |       |                                                   |
  |       v                                                   |
  |  web_agent.invoke({"messages": [("user", query)]})        |
  |       |                                                   |
  |       |   Web Expert内部:                                 |
  |       |   +-------------------------------------------+   |
  |       |   | Agent decides to call web_search         |   |
  |       |   |           |                               |   |
  |       |   |           v                               |   |
  |       |   | web_search.ainvoke(query)                |   |
  |       |   |           |                               |   |
  |       |   |           v                               |   |
  |       |   | DuckDuckGoSearchResults.ainvoke()         |   |
  |       |   |           |                               |   |
  |       |   |           v                               |   |
  |       |   | Returns: "Apple (AAPL) is currently      |   |
  |       |   | trading at $178.50..."                   |   |
  |       |   +-------------------------------------------+   |
  |       |                                                   |
  |       v                                                   |
  |  response["messages"][-1].content                         |
  |       |                                                   |
  |       v                                                   |
  |  Returns: "Apple (AAPL) is currently trading at $178.50"  |
  +-----------------------------------------------------------+

Output: "Apple (AAPL) is currently trading at $178.50..."
```

---

## Data Models (Pydantic)

```
+-------------------------------------------------------------+
|                         INPUT                                |
|                                                             |
|   RAGQueryInput:                                            |
|     query: str           "What is RAG?"                     |
|     session_id: str      "production_user_session_1"        |
|     top_k: int           4                                  |
+-------------------------------------------------------------+
                          |
                          v
+-------------------------------------------------------------+
|                        OUTPUT                                |
|                                                             |
|   RAGQueryOutput:                                           |
|     answer: str          "RAG is Retrieval-Augmented..."    |
|     sources: list[str]   ["data/sample_document.txt"]       |
|     confidence: float    0.85                               |
|                                                             |
|   ToolResponse:                                             |
|     tool_name: str       "search_documents"                 |
|     result: str          "Found 3 relevant passages..."     |
|     status: str          "success"                          |
+-------------------------------------------------------------+
```

---

## Text Normalization Pipeline

```
Raw Document
    |
    v
+---------------------------+
| 1. Unicode normalize      |  NFKD form (accent removal)
|    unicodedata.normalize  |
+-------------+-------------+
              |
              v
+---------------------------+
| 2. Lowercase              |  text.lower()
+-------------+-------------+
              |
              v
+---------------------------+
| 3. Remove special chars   |  Regex: keep \w\s.,;:!?
+-------------+-------------+
              |
              v
+---------------------------+
| 4. Collapse whitespace    |  Regex: \s+ -> " "
+-------------+-------------+
              |
              v
+---------------------------+
| 5. Limit newlines         |  \n{3,} -> \n\n
+-------------+-------------+
              |
              v
    Normalized Document
```

---

## File Responsibility Map

```
+--------------------------------------+------------------------------------------+
| File                                 | Responsibility                           |
+--------------------------------------+------------------------------------------+
| app.py                               | Main entry, async loop, Pydantic         |
|                                      | validation                               |
+--------------------------------------+------------------------------------------+
| config.py                            | Settings (Pydantic BaseSettings)         |
+--------------------------------------+------------------------------------------+
| config_validator.py                  | API key check                            |
+--------------------------------------+------------------------------------------+
| loaders/document_loader.py           | Load PDF/TXT + apply normalization       |
+--------------------------------------+------------------------------------------+
| normalizer/text_normalizer.py        | Unicode, lowercase, whitespace cleanup   |
+--------------------------------------+------------------------------------------+
| chunkers/semantic_chunker.py         | Topic-boundary splitting                 |
+--------------------------------------+------------------------------------------+
| embeddings/embedding_factory.py      | Google embeddings                        |
+--------------------------------------+------------------------------------------+
| embeddings/llm_factory.py            | ChatGoogleGenerativeAI                   |
+--------------------------------------+------------------------------------------+
| vectorstore/faiss_store.py           | FAISS index + retriever                  |
+--------------------------------------+------------------------------------------+
| chains/rag_chain.py                  | LCEL pipeline                            |
|                                      | (retriever|prompt|llm|parser)            |
+--------------------------------------+------------------------------------------+
| tools/search_tool.py                 | Async RAG chain wrapper                  |
+--------------------------------------+------------------------------------------+
| tools/calculator_tool.py             | Async safe math evaluator                |
+--------------------------------------+------------------------------------------+
| tools/web_search_tool.py             | Async DuckDuckGo web search              |
+--------------------------------------+------------------------------------------+
| agents/rag_agent.py                  | Supervisor + 3 worker agents             |
|                                      | (create_agent)                           |
+--------------------------------------+------------------------------------------+
| agents/rag_agent1.py                 | Enhanced variant                         |
+--------------------------------------+------------------------------------------+
| models/schemas.py                    | RAGQueryInput, RAGQueryOutput,           |
|                                      | ToolResponse, AgentState                 |
+--------------------------------------+------------------------------------------+
| prompts/rag_prompt.py                | SYSTEM_PROMPT + RAG_PROMPT template      |
+--------------------------------------+------------------------------------------+
| prompts/agent_prompt.py              | Agent system prompt                      |
+--------------------------------------+------------------------------------------+
| memory/memory_manager.py             | InMemoryChatMessageHistory               |
+--------------------------------------+------------------------------------------+
```

---

## Complete Example: Mixed Query

**User asks:** "Calculate 1250 * 0.18 and summarize the document"

```
1. RAGQueryInput validates input
   - query: "Calculate 1250 * 0.18 and summarize the document"
   - session_id: "production_user_session_1"
   - top_k: 4

2. ainvoke() sends to supervisor

3. Supervisor decides: needs both experts

4. Calls delegate_to_document_expert("summarize the document")
   |
   +---> document_expert
         |
         +---> search_documents.ainvoke()
               |
               +---> rag_chain.ainvoke()
                     |
                     +---> FAISS retriever gets chunks
                     +---> format_docs() joins text
                     +---> Prompt fills context + question
                     +---> LLM generates summary
                     +---> StrOutputParser returns string
         |
         +---> Returns summary to supervisor

5. Calls delegate_to_math_expert("Calculate 1250 * 0.18")
   |
   +---> math_expert
         |
         +---> calculator.ainvoke("1250 * 0.18")
               |
               +---> ast.parse()
               +---> _safe_eval() returns 225.0
         |
         +---> Returns "225.0" to supervisor

6. Supervisor combines both results

7. extract_text() pulls final message

8. "Assistant: The document discusses RAG concepts... The calculation result is 225.0"
```
