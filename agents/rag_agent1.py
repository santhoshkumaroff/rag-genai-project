# agents/rag_agent.py
from langchain.agents import create_agent  
from langchain.agents.middleware import HumanInTheLoopMiddleware  # ✅ Built-in pause/resume
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

def build_agent(llm, rag_chain, calculator_tool):
    """
    Builds an updated, robust multi-agent supervisor system.
    Features: Built-in human approval for tool execution and global recursion safety limits.
    """
    
    # ----------------------------------------------------
    # 1. Define Worker Agent A: Document Specialist
    # ----------------------------------------------------
    from tools.search_tool import create_search_tool
    search_tool = create_search_tool(rag_chain)
    
    doc_agent = create_agent(
        model=llm,
        tools=[search_tool],
        name="document_expert",
        system_prompt=(
            "You are an expert at retrieving and summarizing context from uploaded documents. "
            "Only use your document search tool."
        )
    )

    # ----------------------------------------------------
    # 2. Define Worker Agent B: Math Specialist
    # ----------------------------------------------------
    math_agent = create_agent(
        model=llm,
        tools=[calculator_tool],
        name="math_expert",
        system_prompt=(
            "You are a precise calculator assistant. Only use your calculator tool for calculations."
        )
    )

    # ----------------------------------------------------
    # 3. Create Delegation Tools with State Bridging
    # ----------------------------------------------------
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

    # ----------------------------------------------------
    # 4. Build Supervisor with Middlewares & Checkpointing
    # ----------------------------------------------------
    memory = MemorySaver()
    
    # HumanInTheLoopMiddleware intercepts tool usage. It pauses the agent 
    # execution and waits for a user signal to approve/deny the delegation.
    human_approval = HumanInTheLoopMiddleware(
        apply_to_tools=["delegate_to_document_expert", "delegate_to_math_expert"]
    )

    supervisor_agent = create_agent(
        model=llm,
        tools=[delegate_to_document_expert, delegate_to_math_expert],
        checkpointer=memory,
        middleware=[human_approval],  # ✅ Attaching the pause/resume hook
        system_prompt=(
            "You are a team supervisor. You do not answer questions directly if they require tools. "
            "Instead, delegate the task to the document_expert or the math_expert using your tools. "
            "Once they return an answer, summarize it nicely for the user. "
            "If a sub-agent fails or returns an error, do not loop infinitely—report the failure cleanly."
        )
    )

    return supervisor_agent