# agents/rag_agent.py
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

def build_agent(llm, rag_chain, calculator_tool):
    """
    Builds a multi-agent system where a Supervisor coordinates specialized 
    sub-agents (Doc Agent and Calculator Agent) using prebuilt LangGraph runtimes.
    """
    # 1. Define Worker Agent A: Document Specialist
    # Wrapping your RAG chain directly as a tool for this agent
    from tools.search_tool import create_search_tool
    search_tool = create_search_tool(rag_chain)
    
    doc_agent = create_agent(
        model=llm,
        tools=[search_tool],
        name="document_expert",
        system_prompt="You are an expert at retrieving and summarizing context from uploaded documents. Only use your document tool."
    )

    # 2. Define Worker Agent B: Math Specialist
    math_agent = create_agent(
        model=llm,
        tools=[calculator_tool],
        name="math_expert",
        system_prompt="You are a precise calculator assistant. Only use your calculator tool for calculations."
    )

    # 3. Create Delegation Tools
    # This allows the Supervisor to call the sub-agents as if they were simple functions
    from langchain_core.tools import tool

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

    # 4. Build the Supervisor Agent
    # The supervisor manages conversation memory and directs inputs to the right worker
    memory = MemorySaver()
    supervisor_agent = create_agent(
        model=llm,
        tools=[delegate_to_document_expert, delegate_to_math_expert],
        checkpointer=memory,
        system_prompt=(
            "You are a team supervisor. You do not answer questions directly if they require tools. "
            "Instead, delegate the task to the document_expert or the math_expert using your tools. "
            "Once they return an answer, summarize it nicely for the user."
        )
    )

    return supervisor_agent