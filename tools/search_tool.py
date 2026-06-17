from langchain_core.tools import tool
from models.schemas import ToolResponse


def create_search_tool(rag_chain):

    @tool
    async def search_documents(query: str) -> str:
        """Search information from the document"""
        result = await rag_chain.ainvoke(query)
        response = ToolResponse(
            tool_name="search_documents",
            result=result,
            status="success",
        )
        return response.result

    return search_documents
