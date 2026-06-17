from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool


def build_agent(llm, rag_chain, calculator_tool):
    from tools.search_tool import create_search_tool

    search_tool = create_search_tool(rag_chain)

    doc_agent = create_react_agent(
        model=llm,
        tools=[search_tool],
        prompt="You are an expert at retrieving and summarizing context from uploaded documents.",
    )

    math_agent = create_react_agent(
        model=llm,
        tools=[calculator_tool],
        prompt="You are a precise calculator assistant.",
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

    memory = MemorySaver()

    supervisor_agent = create_react_agent(
        model=llm,
        tools=[delegate_to_document_expert, delegate_to_math_expert],
        checkpointer=memory,
        prompt=(
            "You are a team supervisor. Delegate tasks to document_expert or math_expert. "
            "Summarize their responses for the user."
        ),
    )

    return supervisor_agent
