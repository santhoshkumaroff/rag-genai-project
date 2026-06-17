from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from models.schemas import ToolResponse

duckduckgo_search = DuckDuckGoSearchResults(max_results=3)


@tool
async def web_search(query: str) -> str:
    """Search the web for real-time information using DuckDuckGo"""
    try:
        results = await duckduckgo_search.ainvoke(query)
        response = ToolResponse(
            tool_name="web_search",
            result=results if results else "No results found",
            status="success",
        )
        return response.result
    except Exception as e:
        response = ToolResponse(
            tool_name="web_search",
            result=f"Search failed: {str(e)}",
            status="error",
        )
        return response.result
