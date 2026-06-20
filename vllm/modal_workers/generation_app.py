"""
Modal Serverless GPU Worker - vLLM Text Generation
EXAONE-3.5-2.4B-Instruct 베이스 모델 + 인플루언서별 LoRA 어댑터 추론.

기존 RunPod 워커(vllm/runpod_workers/vllm_generation/generation_worker.py)를
Modal(modal.com) 서버리스로 포팅한 버전.

배포:
    modal deploy vllm/modal_workers/generation_app.py
배포 후 발급되는 URL을 backend .env 의 MODAL_GENERATION_URL 에 설정.

입출력 계약 (백엔드 클라이언트와 일치해야 함):
  입력:  {"input": {"prompt": str,
                    "lora_adapter": str|null,   # = influencer_id (Volume 경로 조회용)
                    "hf_repo": str|null,        # HF LoRA repo (Volume 미존재 시 폴백 다운로드)
                    "hf_token": str|null,       # private repo 다운로드용
                    "system_message": str,
                    "temperature": float,
                    "max_tokens": int}}
  출력:  {"output": {"generated_text": str}}
"""
import logging
import os
from typing import Any, Dict, Optional

import modal

# ---------------------------------------------------------------------------
# 설정 상수
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SYSTEM_MESSAGE = "당신은 도움이 되는 AI 어시스턴트입니다."

# Volume 내 모델/LoRA 캐시 경로
MODELS_DIR = "/models"
LORA_DIR = f"{MODELS_DIR}/lora_adapters"   # /models/lora_adapters/{influencer_id}
HF_CACHE_DIR = f"{MODELS_DIR}/hf_cache"    # 베이스 모델 가중치 캐시

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modal App / Image / Volume
# ---------------------------------------------------------------------------
app = modal.App("aimex-generation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.9.2",
        "torch",
        # transformers 4.54+ 는 aimv2(Ovis)를 native 등록 → vLLM 0.9.2 의 동일 등록과
        # 충돌(ValueError: 'aimv2' is already used). 4.54 미만으로 고정(파인튜닝과 동일).
        "transformers==4.53.1",
        "peft",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]",
    )
    # 베이스 모델 가중치를 Volume에 캐시 (콜드스타트 단축)
    .env({"HF_HOME": HF_CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

# 모델 가중치 / LoRA 어댑터 영구 캐시 볼륨
volume = modal.Volume.from_name("aimex-models", create_if_missing=True)


# ---------------------------------------------------------------------------
# 모델 컨테이너 (vLLM 엔진 싱글톤)
# ---------------------------------------------------------------------------
@app.cls(
    gpu="A10G",
    image=image,
    volumes={MODELS_DIR: volume},
    scaledown_window=60,   # 마지막 요청 후 60초 유지 → scale-to-zero
    timeout=600,
    max_containers=2,
)
class GenerationModel:
    @modal.enter()
    def load(self):
        """컨테이너 시작 시 1회 실행 - vLLM 엔진 + 토크나이저 로드"""
        from transformers import AutoTokenizer
        from vllm import LLM

        logger.info("vLLM 엔진 초기화 시작: %s", DEFAULT_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_MODEL, trust_remote_code=True
        )
        self.engine = LLM(
            model=DEFAULT_MODEL,
            trust_remote_code=True,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=64,
            max_loras=4,  # 7B 베이스 메모리 여유 확보(A10G 24GB) — 동시 LoRA 수 축소
            gpu_memory_utilization=0.92,
            max_model_len=4096,
            enforce_eager=True,
        )
        # influencer_id/hf_repo -> 정수 lora_int_id 매핑 (요청 누적)
        self._lora_ids: Dict[str, int] = {}
        logger.info("vLLM 엔진 초기화 완료")

    # -- 내부 유틸 --------------------------------------------------------
    def _build_prompt(self, user_message: str, system_message: str) -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _resolve_lora_path(
        self,
        lora_adapter: Optional[str],
        hf_repo: Optional[str],
        hf_token: Optional[str],
    ) -> Optional[str]:
        """LoRA 어댑터 로컬 경로 확보.

        1) Volume 경로 /models/lora_adapters/{lora_adapter} 존재 시 사용.
        2) 없으면 hf_repo 에서 다운로드 후 Volume 에 캐시.
        둘 다 없으면 None (베이스 모델 추론).
        """
        # 1) Volume 캐시 우선
        if lora_adapter:
            vol_path = os.path.join(LORA_DIR, lora_adapter)
            if os.path.isdir(vol_path) and os.listdir(vol_path):
                logger.info("Volume LoRA 사용: %s", vol_path)
                return vol_path

        # 2) HF 다운로드 폴백
        if hf_repo:
            from huggingface_hub import snapshot_download

            target = os.path.join(
                LORA_DIR, lora_adapter or hf_repo.replace("/", "__")
            )
            os.makedirs(target, exist_ok=True)
            logger.info("HF LoRA 다운로드: %s -> %s", hf_repo, target)
            snapshot_download(
                repo_id=hf_repo,
                local_dir=target,
                token=hf_token or None,
            )
            volume.commit()  # 다음 콜드스타트에서 재사용되도록 영구화
            return target

        return None

    def _lora_request(self, key: str, path: str):
        from vllm.lora.request import LoRARequest

        if key not in self._lora_ids:
            self._lora_ids[key] = len(self._lora_ids) + 1
        return LoRARequest(
            lora_name=key,
            lora_int_id=self._lora_ids[key],
            lora_path=path,
        )

    # -- 추론 진입점 ------------------------------------------------------
    @modal.method()
    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from vllm import SamplingParams

        prompt_text = payload.get("prompt")
        if not prompt_text:
            raise ValueError("필수 필드 누락: prompt")

        lora_adapter = payload.get("lora_adapter")
        hf_repo = payload.get("hf_repo")
        hf_token = payload.get("hf_token")
        system_message = payload.get("system_message") or DEFAULT_SYSTEM_MESSAGE

        sampling_params = SamplingParams(
            temperature=float(payload.get("temperature", 0.7)),
            max_tokens=int(payload.get("max_tokens", 512)),
            top_p=float(payload.get("top_p", 0.9)),
            top_k=int(payload.get("top_k", 50)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.1)),
        )

        prompt = self._build_prompt(prompt_text, system_message)

        lora_path = self._resolve_lora_path(lora_adapter, hf_repo, hf_token)
        lora_request = None
        if lora_path:
            key = lora_adapter or hf_repo
            lora_request = self._lora_request(key, lora_path)
            logger.info("텍스트 생성 시작 - LoRA: %s", key)
        else:
            logger.info("텍스트 생성 시작 - 베이스 모델")

        outputs = self.engine.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        generated_text = outputs[0].outputs[0].text.strip()
        logger.info("텍스트 생성 완료 (길이: %d)", len(generated_text))
        return {"output": {"generated_text": generated_text}}


# ---------------------------------------------------------------------------
# HTTP 엔드포인트
# ---------------------------------------------------------------------------
@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def generate(item: Dict[str, Any]) -> Dict[str, Any]:
    """POST 엔드포인트.

    백엔드는 {"input": {...}} 형태로 보내거나 평면 dict 로 보낼 수 있다.
    """
    body = item.get("input", item)
    try:
        return GenerationModel().generate.remote(body)
    except Exception as e:  # noqa: BLE001
        logger.error("생성 실패: %s", e)
        return {"output": {"generated_text": ""}, "error": str(e), "status": "failed"}


# ---------------------------------------------------------------------------
# 로컬 테스트: modal run vllm/modal_workers/generation_app.py
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    result = GenerationModel().generate.remote(
        {
            "prompt": "안녕하세요, 자기소개 해주세요.",
            "system_message": DEFAULT_SYSTEM_MESSAGE,
            "temperature": 0.7,
            "max_tokens": 128,
        }
    )
    print(result)
