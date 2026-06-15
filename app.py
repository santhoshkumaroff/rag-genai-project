# app.py
import sys
from config_validator import validate_config
from loaders.document_loader import load_document
from chunkers.semantic_chunker import semantic_chunk_documents
from embeddings.embedding_factory import get_embeddings
from embeddings.llm_factory import get_llm
from vectorstore.faiss_store import create_vector_store, get_retriever
from chains.rag_chain import build_rag_chain
from tools.calculator_tool import calculator

# Imports your newly updated Multi-Agent orchestrator graph
from agents.rag_agent import build_agent


def main():
    # 1. Validate environment configuration strings
    validate_config()

    print("Loading documents...")
    documents = load_document("data/sample_document.txt")

    print("Chunking documents...")
    embeddings = get_embeddings()
    chunks = semantic_chunk_documents(documents, embeddings)

    print("Creating vector store...")
    vectorstore = create_vector_store(chunks, embeddings)
    retriever = get_retriever(vectorstore)

    # 2. Initialize LLM Engine
    llm = get_llm()

    # 3. Assemble Core RAG Processing Logic
    rag_chain = build_rag_chain(retriever, llm)

    # 4. Instantiate Compiled Multi-Agent State Machine
    # We pass components explicitly so the Supervisor can manage delegation
    agent_system = build_agent(llm, rag_chain, calculator)

    print("\n=======================================================")
    print("🚀 Multi-Agent LangGraph RAG System Online")
    print("Specialists Active: [document_expert, math_expert]")
    print("Type 'exit' to quit the application session.")
    print("=======================================================\n")

    # LangGraph uses thread_id configuration blocks to segment sessions
    thread_id = "production_user_session_1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            query = input("User: ").strip()
            
            if not query:
                continue

            if query.lower() == "exit":
                print("Shutting down agent execution loop. Goodbye!")
                break

            # Send prompt down into the compiled graph engine state key ("messages")
            response = agent_system.invoke(
                {
                    "messages": [
                        ("user", query)
                    ]
                },
                config=config,
            )

            # Output processed response string block
            print(f"Assistant: {extract_text(response)}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted via terminal. Exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"An unexpected exception error occurred: {e}\n")


def extract_text(response: dict) -> str:
    """
    Safely navigates the compiled graph's historical state dictionary 
    to extract the string content of the final AI message block.
    """
    if "messages" not in response or not response["messages"]:
        return "Error: No system output state received."

    # Grab the terminal message appended to the state graph list sequence
    final_message = response["messages"][-1]
    content = final_message.content

    # Account for list structures or standard text block outputs
    if isinstance(content, list):
        if len(content) > 0 and isinstance(content[0], dict):
            return content[0].get("text", "")
        return str(content)

    return content


if __name__ == "__main__":
    main()