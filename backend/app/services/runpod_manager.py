"""(레거시) GPU 매니저 진입점 — 현재는 Modal 전용 shim.

과거 RunPod Serverless 구현(BaseRunPodManager / VLLM·TTS·Finetuning RunPodManager 등)을
모두 제거하고, 공개 API(getter·타입·헬스체크·초기화)만 유지해 기존 import 호환성을 보장한다.
실제 GPU 매니저는 app.services.modal_manager 가 제공하며, 본 모듈의 getter 가 그것을 위임 반환한다.

남겨둔 이유: get_vllm_manager / get_tts_manager / get_finetuning_manager 와
HealthStatus·HealthCheckResult 타입이 백엔드 전반(30+ 파일)과 modal_manager 에서 import 되므로,
공개 표면을 보존해 Modal 전환을 무중단으로 유지한다.
"""

import logging
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RunPodManagerError(Exception):
    """(레거시) GPU 관리자 오류 — 호환용 보존."""
    pass


ServiceType = Literal["tts", "vllm", "finetuning"]


class HealthStatus(Enum):
    """Health check 상태"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckResult:
    """Health check 결과 클래스 (modal_manager 가 동일 타입을 사용)."""

    def __init__(
        self,
        status: HealthStatus,
        endpoint_id: Optional[str] = None,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        response_time_ms: Optional[float] = None,
    ):
        self.status = status
        self.endpoint_id = endpoint_id
        self.message = message
        self.details = details or {}
        self.response_time_ms = response_time_ms
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "endpoint_id": self.endpoint_id,
            "message": self.message,
            "details": self.details,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "is_healthy": self.status == HealthStatus.HEALTHY,
        }


# ---------------------------------------------------------------------------
# Modal 매니저 위임 getter (지연 import 로 modal_manager ↔ runpod_manager 순환 회피)
# ---------------------------------------------------------------------------
def get_tts_manager():
    """TTS 매니저 (Modal)."""
    from app.services.modal_manager import get_modal_tts_manager
    return get_modal_tts_manager()


def get_vllm_manager():
    """vLLM(생성/추론) 매니저 (Modal)."""
    from app.services.modal_manager import get_modal_vllm_manager
    return get_modal_vllm_manager()


def get_finetuning_manager():
    """파인튜닝 매니저 (Modal)."""
    from app.services.modal_manager import get_modal_finetuning_manager
    return get_modal_finetuning_manager()


def get_runpod_manager():
    """(레거시 별칭) — TTS 매니저 반환."""
    return get_tts_manager()


def get_manager_by_service_type(service_type: ServiceType):
    """서비스 타입별 매니저 반환 (Modal)."""
    if service_type == "tts":
        return get_tts_manager()
    elif service_type == "vllm":
        return get_vllm_manager()
    elif service_type == "finetuning":
        return get_finetuning_manager()
    raise ValueError(f"지원하지 않는 서비스 타입: {service_type}")


async def health_check_all_services() -> Dict[str, Dict[str, Any]]:
    """모든 GPU 서비스(Modal) health check."""
    results: Dict[str, Any] = {}
    service_types: List[ServiceType] = ["tts", "vllm", "finetuning"]

    for service_type in service_types:
        try:
            manager = get_manager_by_service_type(service_type)
            health_result = await manager.health_check()
            results[service_type] = health_result.to_dict()
        except Exception as e:
            logger.error(f"❌ {service_type} health check 실패: {e}")
            results[service_type] = {
                "status": HealthStatus.UNKNOWN.value,
                "message": f"Health check 중 오류 발생: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "is_healthy": False,
            }

    healthy_count = sum(1 for r in results.values() if r.get("is_healthy", False))
    total_count = len(results)
    results["summary"] = {
        "total_services": total_count,
        "healthy_services": healthy_count,
        "degraded_services": sum(1 for r in results.values() if r.get("status") == "degraded"),
        "unhealthy_services": sum(1 for r in results.values() if r.get("status") in ["unhealthy", "unknown"]),
        "overall_status": "healthy" if healthy_count == total_count else ("degraded" if healthy_count > 0 else "unhealthy"),
        "timestamp": datetime.now().isoformat(),
    }
    return results


async def health_check_service(service_type: ServiceType) -> Dict[str, Any]:
    """특정 GPU 서비스(Modal) health check."""
    try:
        manager = get_manager_by_service_type(service_type)
        health_result = await manager.health_check()
        return health_result.to_dict()
    except Exception as e:
        logger.error(f"❌ {service_type} health check 실패: {e}")
        return {
            "status": HealthStatus.UNKNOWN.value,
            "message": f"Health check 중 오류 발생: {str(e)}",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "is_healthy": False,
        }


async def initialize_runpod():
    """(레거시) 서버 시작 초기화 — Modal 전용 운영이므로 RunPod 초기화는 생략."""
    logger.info("ℹ️ GPU provider=modal — RunPod 비활성(레거시). 초기화 생략.")
