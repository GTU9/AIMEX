"""로그 출력용 민감 정보 마스킹 유틸리티.

payload 등을 로그에 남길 때 hf_token 같은 비밀값이 평문으로 노출되지 않도록
재귀적으로 마스킹한 사본을 만든다. 원본 dict 는 변형하지 않는다(깊은 복사).
"""

import copy
import json
from typing import Any

# 소문자 비교로 매칭. 키 이름에 아래 토큰이 포함되면 마스킹한다(부분 일치).
_SENSITIVE_KEY_PARTS = (
    "hf_token",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "voice_data_base64",
)


def _mask_value(value: str) -> str:
    """앞 4자/뒤 4자만 남기고 마스킹."""
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def _is_sensitive(key: str) -> bool:
    k = key.lower()
    return any(part in k for part in _SENSITIVE_KEY_PARTS)


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and _is_sensitive(k) and isinstance(v, str):
                out[k] = _mask_value(v)
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def redact_secrets(obj: Any) -> Any:
    """민감 키를 마스킹한 깊은 사본을 반환."""
    return _redact(copy.deepcopy(obj))


def safe_json(obj: Any, limit: int = 500) -> str:
    """민감 정보를 마스킹하고 json 문자열로 직렬화(길이 제한)."""
    masked = redact_secrets(obj)
    return json.dumps(masked, ensure_ascii=False, default=str)[:limit]
