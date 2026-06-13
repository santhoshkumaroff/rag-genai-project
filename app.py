from config_validator import validate_config

from loaders.document_loader import load_document
from chunkers.semantic_chunker import semantic_chunk_documents

from embeddings.embedding_factory import get_embeddings
from embeddings.llm_factory import get_llm

from vectorstore.faiss_store import create_vector_store, get_retriever

from chains.rag_chain import build_rag_chain

from tools.search_tool import create_search_tool
from tools.calculator_tool import calculator

from agents.rag_agent import build_agent

from langchain_core.runnables.history import RunnableWithMessageHistory
from memory.memory_manager import get_session_history


def main():

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

    search_tool = create_search_tool(rag_chain)

    tools = [search_tool, calculator]

    agent = build_agent(llm, tools)

    # Memory wrapper
    agent_with_memory = RunnableWithMessageHistory(
        agent,
        get_session_history,
        input_messages_key="messages",
    )

    print("\nRAG Agent Ready\n")

    session_id = "user_1"

    while True:

        query = input("User: ")

        if query.lower() == "exit":
            break

        response = agent_with_memory.invoke(
            {
                "messages": [
                    ("user", query)
                ]
            },
            config={"configurable": {"session_id": session_id}},
        )

        print("Assistant:", extract_text(response))


def extract_text(response):

    content = response["messages"][-1].content

    if isinstance(content, list):
        return content[0].get("text", "")

    return content


if __name__ == "__main__":
    main()