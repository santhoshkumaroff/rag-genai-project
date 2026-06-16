[User Input] 
     │
     ▼
[Supervisor Agent] (Reads prompt)
     │
     ├──► Calls Tool: `delegate_to_document_expert("What was the Q3 revenue and overhead?")`
     │         │
     │         ▼
     │   [Document Expert] ──► Queries FAISS Vector Store ──► Finds context text
     │         │
     │         ▼
     │   Returns text: "Revenue: $50,000, Overhead: $12,000"
     │
     ├──► Calls Tool: `delegate_to_math_expert("Calculate $50,000 minus $12,000")`
     │         │
     │         ▼
     │   [Math Expert] ──► Executes `calculator` tool ──► Computes 38000
     │         │
     │         ▼
     │   Returns text: "38000"
     │
     ▼
[Supervisor Agent] (Summarizes final answer)
     │
     ▼
[extract_text()] ──► Grabs final message content