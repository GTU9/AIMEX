"""
Modal Serverless GPU Worker - Zonos TTS
한국어 음성 합성 + 음성 클로닝(voice cloning).

기존 RunPod 워커(vllm/runpod_workers/tts/tts_worker.py)를 Modal 로 포팅.
zonos 모듈은 runpod_workers/tts/zonos 디렉토리를 그대로 이미지에 포함한다.

배포:
    modal deploy vllm/modal_workers/tts_app.py
배포 후 발급되는 URL을 backend .env 의 MODAL_TTS_URL 에 설정.

입출력 계약 (백엔드 클라이언트와 일치해야 함):
  입력:  {"input": {"text": str,
                    "voice_ref": str|null,    # base64 wav (음성 클로닝 참조 오디오)
                    "language": str,          # 기본 "ko"
                    "emotion_name": str|null, # neutral/happy/sad/... (선택)
                    "speaking_rate": float, "pitch_std": float, "cfg_scale": float}}
  출력:  {"output": {"audio_base64": str, "sample_rate": int, "duration": float}}
"""
import base64
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

# ---------------------------------------------------------------------------
# 설정 상수
# ---------------------------------------------------------------------------
MODELS_DIR = "/models"
HF_CACHE_DIR = f"{MODELS_DIR}/hf_cache"
ZONOS_MODEL = "Zyphra/Zonos-v0.1-transformer"

# 미리 정의된 감정 벡터 (기존 RunPod 워커에서 가져옴)
PREDEFINED_EMOTIONS = {
    "neutral": [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],
    "happy": [0.0256, 0.5897, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.3077],
    "sad": [0.0256, 0.0256, 0.5897, 0.0256, 0.0256, 0.0256, 0.0256, 0.3077],
    "angry": [0.0256, 0.0256, 0.0256, 0.5897, 0.0256, 0.0256, 0.0256, 0.3077],
    "fearful": [0.0256, 0.0256, 0.0256, 0.0256, 0.5897, 0.0256, 0.0256, 0.3077],
    "disgusted": [0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.5897, 0.0256, 0.3077],
    "surprised": [0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.5897, 0.3077],
    "contempt": [0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.8718],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# zonos 모듈 경로 (기존 RunPod 워커 디렉토리 재사용)
_LOCAL_ZONOS = str(Path(__file__).resolve().parent.parent / "runpod_workers" / "tts" / "zonos")

# ---------------------------------------------------------------------------
# Modal App / Image / Volume
# ---------------------------------------------------------------------------
app = modal.App("aimex-tts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # 시스템 의존성: espeak(phonemizer), ffmpeg/libsndfile(torchaudio)
    .apt_install(
        "git",
        "ffmpeg",
        "libsndfile1",
        "espeak",
        "espeak-ng",
        "espeak-ng-data",
        "libespeak-ng1",
        "build-essential",
    )
    .pip_install(
        "torch",
        "torchaudio",
        "transformers>=4.43.0",
        "huggingface_hub",
        "safetensors",
        "inflect",
        "kanjize",
        "phonemizer",
        "sudachipy",
        "sudachidict_full",
        "scipy",
        "numpy",
        "librosa",
        "fastapi[standard]",
    )
    .env({"HF_HOME": HF_CACHE_DIR})
    # 로컬 zonos 패키지를 컨테이너 /root/zonos 로 복사 → import zonos 가능
    .add_local_dir(_LOCAL_ZONOS, remote_path="/root/zonos")
)

volume = modal.Volume.from_name("aimex-models", create_if_missing=True)


# ---------------------------------------------------------------------------
# 모델 컨테이너 (Zonos 엔진 싱글톤)
# ---------------------------------------------------------------------------
@app.cls(
    gpu="A10G",
    image=image,
    volumes={MODELS_DIR: volume},
    scaledown_window=60,
    timeout=600,
    max_containers=2,
)
class TTSModel:
    @modal.enter()
    def load(self):
        """컨테이너 시작 시 1회 - Zonos 모델 로드 (가중치는 Volume에 캐시)"""
        import torch
        from zonos.model import Zonos

        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info("Zonos 모델 초기화 시작 (device=%s)", self.device)
        self.model = Zonos.from_pretrained(ZONOS_MODEL, device=self.device)
        # 다운로드된 HF 가중치를 Volume에 영구화 (다음 콜드스타트 단축)
        volume.commit()
        logger.info("Zonos 모델 초기화 완료")

    # -- 내부 유틸 --------------------------------------------------------
    def _validate(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        if not job_input.get("text"):
            raise ValueError("text 필드는 필수입니다.")

        emotion = job_input.get("emotion", PREDEFINED_EMOTIONS["neutral"])
        emotion_name = job_input.get("emotion_name")
        if emotion_name and emotion_name in PREDEFINED_EMOTIONS:
            emotion = PREDEFINED_EMOTIONS[emotion_name]
        if len(emotion) != 8 or not all(0 <= x <= 1 for x in emotion):
            raise ValueError("emotion은 0~1 범위의 float 8개여야 합니다.")

        # voice_ref(신규 계약) 또는 voice_data_base64(기존 계약) 모두 허용
        voice_ref = job_input.get("voice_ref") or job_input.get("voice_data_base64")

        return {
            "text": job_input["text"],
            "language": "ko",  # 한국어 고정 (기존 워커 동작 유지)
            "speaking_rate": float(job_input.get("speaking_rate", 22.0)),
            "pitch_std": float(job_input.get("pitch_std", 40.0)),
            "cfg_scale": float(job_input.get("cfg_scale", 4.0)),
            "emotion": emotion,
            "voice_ref": voice_ref,
        }

    def _speaker_embedding(self, voice_ref_b64: str):
        import torchaudio

        voice_data = base64.b64decode(voice_ref_b64)
        voice_wav, sr = torchaudio.load(io.BytesIO(voice_data))
        voice_wav = voice_wav.to(self.device)
        return self.model.make_speaker_embedding(voice_wav, sr)

    def _synthesize(
        self,
        text: str,
        speaker_embedding,
        language: str,
        speaking_rate: float,
        pitch_std: float,
        cfg_scale: float,
        emotion: List[float],
    ):
        from zonos.conditioning import make_cond_dict

        cond_dict = make_cond_dict(
            text=text,
            speaker=speaker_embedding,
            language=language,
            speaking_rate=speaking_rate,
            emotion=emotion,
            pitch_std=pitch_std,
            device=self.device,
        )
        conditioning = self.model.prepare_conditioning(cond_dict)
        with self.torch.no_grad():
            codes = self.model.generate(
                conditioning,
                cfg_scale=cfg_scale,
                disable_torch_compile=True,
                progress_bar=False,
            )
        wavs = self.model.autoencoder.decode(codes)
        return wavs[0]

    # -- 추론 진입점 ------------------------------------------------------
    @modal.method()
    def synthesize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import torchaudio

        v = self._validate(payload)
        logger.info("TTS 생성 - 텍스트 길이: %d", len(v["text"]))

        speaker_embedding = None
        if v["voice_ref"]:
            logger.info("음성 클로닝 처리")
            speaker_embedding = self._speaker_embedding(v["voice_ref"])

        wav_output = self._synthesize(
            text=v["text"],
            speaker_embedding=speaker_embedding,
            language=v["language"],
            speaking_rate=v["speaking_rate"],
            pitch_std=v["pitch_std"],
            cfg_scale=v["cfg_scale"],
            emotion=v["emotion"],
        )

        sample_rate = self.model.autoencoder.sampling_rate
        wav_cpu = wav_output.cpu()
        audio_buffer = io.BytesIO()
        torchaudio.save(audio_buffer, wav_cpu, sample_rate, format="wav")
        audio_buffer.seek(0)
        audio_data = audio_buffer.getvalue()
        duration = float(wav_cpu.shape[1]) / sample_rate

        logger.info("TTS 생성 완료 (%.2fs, %d bytes)", duration, len(audio_data))
        return {
            "output": {
                "audio_base64": base64.b64encode(audio_data).decode(),
                "sample_rate": sample_rate,
                "duration": duration,
            }
        }


# ---------------------------------------------------------------------------
# HTTP 엔드포인트
# ---------------------------------------------------------------------------
@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def tts(item: Dict[str, Any]) -> Dict[str, Any]:
    body = item.get("input", item)
    try:
        return TTSModel().synthesize.remote(body)
    except Exception as e:  # noqa: BLE001
        logger.error("TTS 실패: %s", e)
        return {"output": {"audio_base64": ""}, "error": str(e), "status": "failed"}


# ---------------------------------------------------------------------------
# 로컬 테스트: modal run vllm/modal_workers/tts_app.py
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    result = TTSModel().synthesize.remote(
        {"text": "안녕하세요, 모달 음성 합성 테스트입니다.", "emotion_name": "happy"}
    )
    out = result.get("output", {})
    print(
        f"audio_base64 len={len(out.get('audio_base64', ''))} "
        f"sample_rate={out.get('sample_rate')} duration={out.get('duration')}"
    )
