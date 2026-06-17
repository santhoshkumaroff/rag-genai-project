import asyncio
import sys
from config_validator import validate_config
from loaders.document_loader import load_document
from chunkers.semantic_chunker import semantic_chunk_documents
from embeddings.embedding_factory import get_embeddings
from embeddings.llm_factory import get_llm
from vectorstore.faiss_store import create_vector_store, get_retriever
from chains.rag_chain import build_rag_chain
from tools.calculator_tool import calculator
from agents.rag_agent import build_agent
from models.schemas import RAGQueryInput


def extract_text(response: dict) -> str:
    if "messages" not in response or not response["messages"]:
        return "Error: No system output state received."

    final_message = response["messages"][-1]
    content = final_message.content

    if isinstance(content, list):
        if len(content) > 0 and isinstance(content[0], dict):
            return content[0].get("text", "")
        return str(content)

    return content


async def main():
    validate_config()

    print("Loading documents...")
    documents = load_document("data/sample_document.txt")

    print("Chunking documents...")
    embeddings = get_embeddings()
    chunks = semantic_chunk_documents(documents, embeddings)

    print("Creating vector store...")
    vectorstore = create_vector_store(chunks, embeddings)
    retriever = get_retriever(vectorstore)

    llm = get_llm()
    rag_chain = build_rag_chain(retriever, llm)
    agent_system = build_agent(llm, rag_chain, calculator)

    print("\n=======================================================")
    print("Multi-Agent LangGraph RAG System Online")
    print("Specialists Active: [document_expert, math_expert, web_expert]")
    print("Type 'exit' to quit the application session.")
    print("=======================================================\n")

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

            query_input = RAGQueryInput(query=query, session_id=thread_id)

            response = await agent_system.ainvoke(
                {"messages": [("user", query_input.query)]},
                config=config,
            )

            print(f"Assistant: {extract_text(response)}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted via terminal. Exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"An unexpected exception error occurred: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
