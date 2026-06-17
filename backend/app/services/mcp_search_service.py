"""MCP 검색 보강 서비스.

인플루언서에 할당된 MCP 서버(웹검색 등)를 호출해, 문서(RAG)에 없는 정보를
외부 검색으로 보완한다. 외부 회사 MCP 서버도 동일 인터페이스(등록→할당)로 동작.

설계: chatbot 에서 RAG 컨텍스트가 없을 때만 호출(문서 우선, 검색은 보완).
실패 시 ('', []) 폴백하여 챗봇을 절대 중단시키지 않는다.
"""

import json
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def _extract_text(result) -> str:
    """langchain MCP 도구 결과를 평문으로 변환.

    결과는 str 이거나 [{'type':'text','text':...}] 형태일 수 있다.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(result, dict) and result.get("text"):
        return result["text"]
    return str(result) if result else ""


def _build_client_config(servers) -> Dict[str, dict]:
    """할당된 MCP 서버 row 들로 MultiServerMCPClient 설정 구성."""
    from app.services.mcp_server_manager import get_command_path

    config: Dict[str, dict] = {}
    for s in servers:
        try:
            cfg = json.loads(s.mcp_config)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[MCP] '{s.mcp_name}' config 파싱 실패: {e}")
            continue
        if "command" in cfg and "args" in cfg:
            config[s.mcp_name] = {
                "command": get_command_path(cfg["command"]),
                "args": cfg["args"],
                "transport": "stdio",
            }
        elif "url" in cfg:
            config[s.mcp_name] = {"url": cfg["url"], "transport": "streamable_http"}
    return config


async def search_via_assigned_mcp(influencer_id: str, query: str, db) -> Tuple[str, List[Dict]]:
    """인플루언서에 할당된 MCP 서버의 검색 도구를 호출해 컨텍스트를 반환.

    반환: (context_str, sources). 할당 없음/실패/0건 시 ('', []).
    """
    if not influencer_id or not query:
        return "", []
    try:
        from app.services.mcp_server_service import MCPServerService

        servers = MCPServerService(db).get_influencer_mcp_servers(influencer_id)
        if not servers:
            return "", []

        config = _build_client_config(servers)
        if not config:
            return "", []

        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        client = MultiServerMCPClient(config)
        for name in config:
            try:
                async with client.session(name) as session:
                    tools = await load_mcp_tools(session)
                    if not tools:
                        continue
                    # 'search' 가 포함된 도구 우선, 없으면 첫 도구
                    tool = next(
                        (t for t in tools if "search" in t.name.lower()), tools[0]
                    )
                    result = await tool.ainvoke({"query": query})
                    text = _extract_text(result).strip()
                    if text and "검색 실패" not in text and "결과가 없습니다" not in text:
                        logger.info(f"[MCP] '{name}.{tool.name}' 검색 보강 성공 ({len(text)}자)")
                        sources = [{"text": text[:200], "source": f"{name}:{tool.name}", "type": "mcp"}]
                        return text, sources
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[MCP] 서버 '{name}' 호출 실패: {e}")
                continue
        return "", []
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[MCP] 검색 보강 실패(폴백): {e}")
        return "", []
