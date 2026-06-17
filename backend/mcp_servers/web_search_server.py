"""무료 웹검색 MCP 서버 (stdio).

API 키 없이 DuckDuckGo(ddgs)로 웹검색을 제공하는 MCP 서버.
챗봇이 "문서에 없는 정보"를 답할 때 이 도구를 호출해 최신/외부 정보를 가져온다.

외부 회사 MCP 서버도 동일한 방식(다른 command/url)으로 등록만 하면 대체된다.

실행: python backend/mcp_servers/web_search_server.py   (stdio transport)
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web-search")


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """웹에서 정보를 검색한다. 문서에 없는 최신/외부 정보가 필요할 때 사용.

    Args:
        query: 검색어
        max_results: 가져올 결과 수 (기본 5)
    Returns:
        제목/요약/링크가 정리된 검색 결과 텍스트
    """
    try:
        from ddgs import DDGS

        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:  # noqa: BLE001
        return f"검색 실패: {e}"

    if not results:
        return "검색 결과가 없습니다."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   출처: {href}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
