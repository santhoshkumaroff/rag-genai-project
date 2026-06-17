from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool


def build_agent(llm, rag_chain, calculator_tool):
    from tools.search_tool import create_search_tool
    from tools.web_search_tool import web_search

    search_tool = create_search_tool(rag_chain)

    doc_agent = create_agent(
        model=llm,
        tools=[search_tool, web_search],
        name="document_expert",
        system_prompt=(
            "You are an expert at retrieving and summarizing context. "
            "Use the search_documents tool for local document queries. "
            "Use the web_search tool as a fallback when the document does not contain the answer."
        ),
    )

    math_agent = create_agent(
        model=llm,
        tools=[calculator_tool],
        name="math_expert",
        system_prompt=(
            "You are a precise calculator assistant. "
            "Only use your calculator tool for calculations."
        ),
    )

    web_agent = create_agent(
        model=llm,
        tools=[web_search],
        name="web_expert",
        system_prompt=(
            "You are a web research specialist. "
            "Use the web_search tool to find real-time information from the internet."
        ),
    )

    @tool
    def delegate_to_document_expert(query: str) -> str:
        """Use this tool when the user asks questions about documents, files, data summaries, or textual topics."""
        response = doc_agent.invoke({"messages": [("user", query)]})
        return response["messages"][-1].content

    @tool
    def delegate_to_math_expert(query: str) -> str:
        """Use this tool when the user needs math operations, equations, formulas, or arithmetic calculations."""
        response = math_agent.invoke({"messages": [("user", query)]})
        return response["messages"][-1].content

    @tool
    def delegate_to_web_expert(query: str) -> str:
        """Use this tool when the user asks about current events, real-time data, or topics not covered in local documents."""
        response = web_agent.invoke({"messages": [("user", query)]})
        return response["messages"][-1].content

    memory = MemorySaver()

    supervisor_agent = create_agent(
        model=llm,
        tools=[delegate_to_document_expert, delegate_to_math_expert, delegate_to_web_expert],
        checkpointer=memory,
        system_prompt=(
            "You are a team supervisor. You do not answer questions directly if they require tools. "
            "Delegate tasks using your tools:\n"
            "- delegate_to_document_expert for document/file questions\n"
            "- delegate_to_math_expert for math calculations\n"
            "- delegate_to_web_expert for current events or real-time information\n"
            "Once they return an answer, summarize it nicely for the user."
        ),
    )

    return supervisor_agent
