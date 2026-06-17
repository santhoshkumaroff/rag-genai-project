from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from models.schemas import ToolResponse


tavily_search = TavilySearchResults(max_results=3)


@tool
async def web_search(query: str) -> str:
    """Search the web for real-time information using Tavily"""
    try:
        results = await tavily_search.ainvoke(query)
        formatted = "\n\n".join(
            f"**{r.get('title', 'No title')}**\n{r.get('content', '')}"
            for r in results
        )
        response = ToolResponse(
            tool_name="web_search",
            result=formatted if formatted else "No results found",
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
