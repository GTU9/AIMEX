"""(레거시) RunPod GPU Pod 서비스 — 현재는 Modal 전용 운영으로 비활성화된 no-op stub.

과거 RunPod GraphQL API 로 ComfyUI Pod 를 생성/관리하던 구현(~1300줄)을 전부 제거했다.
이미지 생성/수정은 Modal 서버리스(/modal-generate, /modal-modify)로 대체되었으므로
RunPod Pod 는 더 이상 생성하지 않는다.

호환성을 위해 공개 표면(RunPodPodRequest/RunPodPodResponse/RunPodService/get_runpod_service)은
유지하되, 모든 Pod 동작은 "비활성(unavailable)" 을 반환한다. 이를 호출하는 레거시 엔드포인트
(/generate, /modify-simple, /synthesize 등)는 endpoint_url 부재로 깔끔히 '사용 불가'를 반환한다.
"""

import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_UNAVAILABLE_MSG = "RunPod Pod 는 비활성 상태입니다 (Modal 전용 운영). 이미지 생성은 /modal-generate 를 사용하세요."


class RunPodPodRequest(BaseModel):
    """(레거시) RunPod 인스턴스 생성 요청 — 호환용 보존."""
    name: str = ""
    template_id: str = ""
    gpu_type: str = "NVIDIA RTX A6000"
    gpu_count: int = 1
    container_disk_in_gb: int = 20
    volume_in_gb: int = 0
    ports: str = "8188/http"
    env: Dict[str, str] = {}


class RunPodPodResponse(BaseModel):
    """(레거시) RunPod 인스턴스 정보 — 호환용 보존."""
    pod_id: str
    status: str  # STARTING, RUNNING, STOPPED, FAILED
    runtime: Optional[Dict[str, Any]] = None
    endpoint_url: Optional[str] = None
    cost_per_hour: Optional[float] = None


class RunPodService:
    """(레거시) RunPod 서비스 — 모든 Pod 동작이 비활성(no-op)."""

    def __init__(self):
        # RunPod 자격증명 불필요 — Modal 전용 운영
        logger.debug("RunPodService(no-op) 초기화 — RunPod 비활성, Modal 전용")

    def _generate_proxy_url(self, pod_id: str, internal_port: int = 8188) -> str:
        return ""

    async def create_pod(self, request_id: str) -> RunPodPodResponse:
        logger.info(f"ℹ️ create_pod 무시(no-op): {_UNAVAILABLE_MSG}")
        return RunPodPodResponse(pod_id="modal-noop", status="STOPPED", endpoint_url=None)

    async def get_pod_status(self, pod_id: str) -> RunPodPodResponse:
        return RunPodPodResponse(pod_id=pod_id or "modal-noop", status="STOPPED",
                                 runtime=None, endpoint_url=None)

    async def terminate_pod(self, pod_id: str) -> bool:
        return True

    async def wait_for_ready(self, pod_id: str, max_wait_time: int = 600) -> bool:
        return False

    async def _check_comfyui_ready(self, endpoint_url: str, max_retries: int = 3,
                                   retry_delay: int = 3) -> bool:
        return False

    async def check_pod_health(self, pod_id: str) -> dict:
        return {"healthy": False, "error": _UNAVAILABLE_MSG}

    async def force_restart_pod(self, pod_id: str, request_id: str) -> dict:
        return {"success": False, "message": _UNAVAILABLE_MSG}

    async def check_volume_status(self, volume_id: str = None) -> dict:
        return {"error": _UNAVAILABLE_MSG}

    async def get_remaining_credits(self) -> Optional[Dict[str, Any]]:
        return None


_runpod_service: Optional[RunPodService] = None


def get_runpod_service() -> RunPodService:
    global _runpod_service
    if _runpod_service is None:
        _runpod_service = RunPodService()
    return _runpod_service
