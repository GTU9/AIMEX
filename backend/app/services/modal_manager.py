"""
Modal Serverless GPU 관리 서비스

RunPod 매니저(runpod_manager.py)와 동일한 메서드 시그니처/반환형을 갖는
Modal 기반 추론 매니저들을 제공한다. 호출부(chat.py 등)는 RunPod 매니저와
Modal 매니저를 구분하지 않고 동일하게 사용할 수 있어야 한다.

Modal 함수는 RunPod 워커와 동일한 입출력 계약을 따른다(계약):
  - 입력: {"input": {"prompt", "lora_adapter", "system_message", "temperature", "max_tokens", ...}}
  - 출력: {"output": {"generated_text": "..."}}

URL은 settings.MODAL_GENERATION_URL / MODAL_TTS_URL / MODAL_FINETUNING_URL 에서 읽는다.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings
# HealthCheck 관련 타입은 RunPod 매니저 것을 재사용한다.
from app.services.runpod_manager import HealthCheckResult, HealthStatus

logger = logging.getLogger(__name__)


class ModalManagerError(Exception):
    """Modal 관리자 오류"""
    pass


class BaseModalManager:
    """Modal 엔드포인트 관리자 베이스 클래스

    RunPod 매니저와 달리 엔드포인트 탐색이 필요 없고, 설정된 URL로 직접 POST 한다.
    """

    def __init__(self, url: Optional[str], service_name: str):
        self._url = url
        self._service_name = service_name

    @property
    def headers(self) -> Dict[str, str]:
        """API 요청 헤더

        MODAL_AUTH_TOKEN 이 설정된 경우 인증 헤더를 추가한다.
        우선 `Authorization: Bearer {token}` 방식을 사용한다.
        대안: Modal proxy auth 분리 토큰을 쓰는 경우
              {"Modal-Key": <id>, "Modal-Secret": <secret>} 또는
              {"Token-Id": <id>, "Token-Secret": <secret>} 형태로 교체한다.
        """
        headers = {"Content-Type": "application/json"}
        if settings.MODAL_AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {settings.MODAL_AUTH_TOKEN}"
        return headers

    def _require_url(self) -> str:
        if not self._url:
            raise ModalManagerError(
                f"{self._service_name} Modal URL이 설정되지 않았습니다 "
                f"(GPU_PROVIDER=modal 사용 시 해당 MODAL_*_URL 환경변수를 설정하세요)"
            )
        return self._url

    async def find_endpoint(self) -> Optional[Dict[str, Any]]:
        """RunPod 매니저 호환용. Modal 은 고정 URL 기반이라 엔드포인트 탐색이
        불필요하므로, URL 이 설정돼 있으면 의사 엔드포인트 정보를 반환한다."""
        return {"id": self._url, "url": self._url} if self._url else None

    async def get_or_create_endpoint(self) -> Optional[Dict[str, Any]]:
        """RunPod 매니저 호환용. Modal 은 생성 과정이 없어 find_endpoint 와 동일."""
        return await self.find_endpoint()

    async def health_check(self) -> HealthCheckResult:
        """Modal 엔드포인트 health check

        Modal fastapi_endpoint는 GET 미지원일 수 있으므로, URL 도달 가능 여부만
        가볍게 확인한다. 어떤 응답이든 받으면 HEALTHY, 예외/미설정이면 UNHEALTHY.
        예외는 전파하지 않고 항상 HealthCheckResult로만 반환한다.
        """
        if not self._url:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                endpoint_id=None,
                message=f"{self._service_name} Modal URL이 설정되지 않았습니다",
            )

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # 가벼운 GET 시도. Modal endpoint가 GET을 막아 4xx/405를 줘도
                # 서버가 살아있다는 신호이므로 HEALTHY로 본다.
                response = await client.get(self._url)
            elapsed_ms = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                endpoint_id=self._url,
                message=f"{self._service_name} Modal 엔드포인트 응답 확인 (status={response.status_code})",
                details={"http_status": response.status_code},
                response_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning(f"⚠️ {self._service_name} Modal health check 실패: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                endpoint_id=self._url,
                message=f"{self._service_name} Modal 엔드포인트 연결 실패: {e}",
                response_time_ms=elapsed_ms,
            )

    async def simple_health_check(self) -> bool:
        """간단한 health check (RunPod 매니저 호환)"""
        try:
            result = await self.health_check()
            return result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        except Exception as e:
            logger.warning(f"⚠️ {self._service_name} Modal 상태 확인 실패: {e}")
            return False


class ModalVLLMManager(BaseModalManager):
    """vLLM 서비스용 Modal 매니저 (VLLMRunPodManager 호환)"""

    def __init__(self):
        super().__init__(settings.MODAL_GENERATION_URL, "vLLM")

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """비동기 요청 (Modal은 동기 호출이므로 runsync와 동일 결과 반환)"""
        url = self._require_url()

        logger.info(f"🚀 Modal run 요청: {url}")

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, headers=self.headers, json=payload)

            if response.status_code != 200:
                error_msg = f"Modal API 오류: {response.status_code} - {response.text}"
                logger.error(f"❌ {error_msg}")
                raise ModalManagerError(error_msg)

            return response.json()

    async def runsync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """동기 요청 (결과 대기)

        RunPod VLLMRunPodManager.runsync 와 동일한 응답 파싱 로직을 따른다.
        """
        url = self._require_url()

        logger.info(f"⏳ Modal runsync 요청: {url}")
        from app.utils.log_redaction import safe_json
        logger.info(f"📦 Payload: {safe_json(payload)}...")

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(url, headers=self.headers, json=payload)

                logger.info(f"📡 Response status: {response.status_code}")

                if response.status_code != 200:
                    error_msg = f"Modal API 오류: {response.status_code} - {response.text}"
                    logger.error(f"❌ {error_msg}")
                    raise ModalManagerError(error_msg)

                result = response.json()
                logger.info(f"✅ Modal response: {json.dumps(result, ensure_ascii=False)[:500]}...")

                # Modal 응답에서 실제 텍스트 추출 (RunPod runsync 파싱 로직 동일)
                if "output" in result:
                    # output이 문자열인 경우
                    if isinstance(result["output"], str):
                        return result["output"]["generated_text"]
                    # output이 딕셔너리인 경우
                    elif isinstance(result["output"], dict):
                        return result["output"].get("generated_text", result["output"])
                    else:
                        return result["output"]
                else:
                    logger.warning(f"⚠️ Unexpected response format: {result}")
                    return result

        except httpx.TimeoutException as e:
            logger.error(f"❌ Modal 요청 타임아웃: {e}")
            raise ModalManagerError("Modal 요청 타임아웃 (300초 초과)")
        except ModalManagerError:
            raise
        except Exception as e:
            logger.error(f"❌ Modal runsync 실패: {e}")
            raise ModalManagerError(f"Modal runsync 실패: {e}")

    async def stream(self, payload: Dict[str, Any]):
        """스트리밍 요청 (async generator)

        Modal 함수가 SSE(text/event-stream)를 반환한다는 동일 계약을 따른다.
        """
        url = self._require_url()

        stream_headers = dict(self.headers)
        stream_headers.update({
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        })

        logger.info(f"🌊 Modal stream 요청: {url}")

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", url, headers=stream_headers, json=payload) as response:
                    if response.status_code != 200:
                        error_msg = f"Modal API 오류: {response.status_code} - {await response.aread()}"
                        logger.error(f"❌ {error_msg}")
                        raise ModalManagerError(error_msg)

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip():
                                try:
                                    data = json.loads(data_str)
                                    yield data
                                except json.JSONDecodeError:
                                    continue
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Modal 스트리밍 요청 실패: {e}")
            raise ModalManagerError(f"Modal 스트리밍 요청 실패: {e}")


class ModalTTSManager(BaseModalManager):
    """TTS 서비스용 Modal 매니저 (TTSRunPodManager 호환)"""

    def __init__(self):
        super().__init__(settings.MODAL_TTS_URL, "TTS")

    async def runsync(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """동기 TTS 음성 생성 (결과 대기)

        RunPod TTS 매니저와 달리 페이로드는 Modal 함수가 그대로 처리한다는
        동일 계약을 따른다. 입력이 {"input": {...}} 형태가 아니면 감싼다.
        """
        url = self._require_url()

        # {"input": {...}} 형태로 표준화 (RunPod 계약과 동일)
        if "input" in job_input:
            payload = job_input
        else:
            payload = {"input": job_input}

        logger.info(f"⏳ Modal TTS runsync 요청: {url}")

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(url, headers=self.headers, json=payload)

                if response.status_code != 200:
                    error_msg = f"Modal TTS API 오류: {response.status_code} - {response.text}"
                    logger.error(f"❌ {error_msg}")
                    raise ModalManagerError(error_msg)

                return response.json()

        except httpx.TimeoutException as e:
            logger.error(f"❌ Modal TTS 요청 타임아웃: {e}")
            raise ModalManagerError("Modal TTS 요청 타임아웃 (300초 초과)")
        except ModalManagerError:
            raise
        except Exception as e:
            logger.error(f"❌ Modal TTS runsync 실패: {e}")
            raise ModalManagerError(f"Modal TTS runsync 실패: {e}")


class ModalImageManager(BaseModalManager):
    """이미지 생성(SDXL-Turbo) Modal 매니저

    텍스트→이미지 생성. 입력이 {"input": {...}} 형태가 아니면 감싼다.
    반환은 Modal 응답 그대로({"output": {"image_base64", "width", "height", "seed"}}).
    """

    def __init__(self):
        super().__init__(settings.MODAL_IMAGE_URL, "Image")

    async def runsync(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """동기 이미지 생성 (결과 대기)"""
        url = self._require_url()

        # {"input": {...}} 형태로 표준화
        if "input" in job_input:
            payload = job_input
        else:
            payload = {"input": job_input}

        logger.info(f"⏳ Modal Image runsync 요청: {url}")

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(url, headers=self.headers, json=payload)

                if response.status_code != 200:
                    error_msg = f"Modal Image API 오류: {response.status_code} - {response.text}"
                    logger.error(f"❌ {error_msg}")
                    raise ModalManagerError(error_msg)

                return response.json()

        except httpx.TimeoutException as e:
            logger.error(f"❌ Modal Image 요청 타임아웃: {e}")
            raise ModalManagerError("Modal Image 요청 타임아웃 (300초 초과)")
        except ModalManagerError:
            raise
        except Exception as e:
            logger.error(f"❌ Modal Image runsync 실패: {e}")
            raise ModalManagerError(f"Modal Image runsync 실패: {e}")


class ModalFinetuningManager(BaseModalManager):
    """Fine-tuning 서비스용 Modal 매니저 (FinetuningRunPodManager 호환)"""

    def __init__(self):
        super().__init__(settings.MODAL_FINETUNING_URL, "Finetuning")

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """비동기 요청"""
        url = self._require_url()

        logger.info(f"🚀 Modal Finetuning run 요청: {url}")

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, headers=self.headers, json=payload)

            if response.status_code != 200:
                error_msg = f"Modal Finetuning API 오류: {response.status_code} - {response.text}"
                logger.error(f"❌ {error_msg}")
                raise ModalManagerError(error_msg)

            return response.json()

    async def runsync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """동기 요청 (결과 대기)"""
        url = self._require_url()

        logger.info(f"⏳ Modal Finetuning runsync 요청: {url}")

        try:
            # 파인튜닝은 장시간 작업(수십 분~). 동기 호출이므로 긴 timeout 사용.
            # (정석은 Modal spawn + 상태 폴링이나, 현재는 백그라운드 task 내 동기 호출.)
            async with httpx.AsyncClient(timeout=1800) as client:
                response = await client.post(url, headers=self.headers, json=payload)

                if response.status_code != 200:
                    error_msg = f"Modal Finetuning API 오류: {response.status_code} - {response.text}"
                    logger.error(f"❌ {error_msg}")
                    raise ModalManagerError(error_msg)

                return response.json()

        except httpx.TimeoutException as e:
            logger.error(f"❌ Modal Finetuning 요청 타임아웃: {e}")
            raise ModalManagerError("Modal Finetuning 요청 타임아웃 (300초 초과)")
        except ModalManagerError:
            raise
        except Exception as e:
            logger.error(f"❌ Modal Finetuning runsync 실패: {e}")
            raise ModalManagerError(f"Modal Finetuning runsync 실패: {e}")


# 싱글톤 인스턴스들
_modal_vllm_manager: Optional[ModalVLLMManager] = None
_modal_tts_manager: Optional[ModalTTSManager] = None
_modal_finetuning_manager: Optional[ModalFinetuningManager] = None
_modal_image_manager: Optional["ModalImageManager"] = None


def get_modal_vllm_manager() -> ModalVLLMManager:
    """Modal vLLM 매니저 싱글톤 인스턴스 반환"""
    global _modal_vllm_manager
    if _modal_vllm_manager is None:
        _modal_vllm_manager = ModalVLLMManager()
    return _modal_vllm_manager


def get_modal_tts_manager() -> ModalTTSManager:
    """Modal TTS 매니저 싱글톤 인스턴스 반환"""
    global _modal_tts_manager
    if _modal_tts_manager is None:
        _modal_tts_manager = ModalTTSManager()
    return _modal_tts_manager


def get_modal_finetuning_manager() -> ModalFinetuningManager:
    """Modal Fine-tuning 매니저 싱글톤 인스턴스 반환"""
    global _modal_finetuning_manager
    if _modal_finetuning_manager is None:
        _modal_finetuning_manager = ModalFinetuningManager()
    return _modal_finetuning_manager


def get_modal_image_manager() -> "ModalImageManager":
    """Modal 이미지 생성 매니저 싱글톤 인스턴스 반환"""
    global _modal_image_manager
    if _modal_image_manager is None:
        _modal_image_manager = ModalImageManager()
    return _modal_image_manager
