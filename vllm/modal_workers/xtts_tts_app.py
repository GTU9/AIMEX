"""Modal Serverless GPU Worker - XTTS-v2 TTS (Zonos/CosyVoice 대체).

한국어 음성 합성 + 제로샷 음성 클로닝(coqui-tts / XTTS-v2).
- 베이스 음성(voice_ref) 있으면 그 음색으로 클로닝
- 없으면 XTTS-v2 내장 스피커로 합성

배포:
    modal deploy vllm/modal_workers/xtts_tts_app.py
배포 후 URL을 backend .env 의 MODAL_TTS_URL 에 설정.

입출력 계약 (기존 aimex-tts 와 동일):
  입력:  {"input": {"text": str, "voice_ref": str|null, "voice_data_base64": str|null, "language": str}}
  출력:  {"output": {"audio_base64": str, "sample_rate": int, "duration": float}}
"""
import base64
import io
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = modal.App("aimex-xtts")

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


def _download_model():
    os.environ["COQUI_TOS_AGREED"] = "1"
    from TTS.api import TTS

    TTS(MODEL_NAME)  # 모델 가중치 다운로드(이미지에 베이크)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch", "torchaudio", "torchcodec", "transformers==4.46.3", "coqui-tts", "fastapi[standard]", "soundfile", "hangul_romanize", "jamo")
    .env({"COQUI_TOS_AGREED": "1"})
    .run_function(_download_model)
)


@app.cls(gpu="A10G", image=image, scaledown_window=120, timeout=600, max_containers=2)
class XTTS:
    @modal.enter()
    def load(self):
        os.environ["COQUI_TOS_AGREED"] = "1"
        import torch
        from TTS.api import TTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("XTTS-v2 로드 시작 (device=%s)", device)
        self.tts = TTS(MODEL_NAME).to(device)
        self.sample_rate = int(getattr(self.tts.synthesizer, "output_sample_rate", 24000))
        # 내장 스피커(베이스 음성 없을 때 폴백)
        try:
            names = list(self.tts.synthesizer.tts_model.speaker_manager.name_to_id.keys())
            self.default_speaker = names[0] if names else None
        except Exception:  # noqa: BLE001
            self.default_speaker = None
        logger.info("XTTS-v2 로드 완료 (sr=%d, default_speaker=%s)", self.sample_rate, self.default_speaker)

    @modal.method()
    def synthesize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import soundfile as sf

        text = payload.get("text")
        if not text:
            raise ValueError("text 필드는 필수입니다.")
        language = payload.get("language", "ko")
        voice_ref = payload.get("voice_ref") or payload.get("voice_data_base64")

        tmp_ref = None
        try:
            if voice_ref:
                data = base64.b64decode(voice_ref)
                tmp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp_ref.write(data)
                tmp_ref.close()
                logger.info("음성 클로닝 모드 (참조 %d bytes)", len(data))
                wav = self.tts.tts(text=text, speaker_wav=tmp_ref.name, language=language)
            else:
                logger.info("내장 스피커 모드 (speaker=%s)", self.default_speaker)
                wav = self.tts.tts(text=text, speaker=self.default_speaker, language=language)
        finally:
            if tmp_ref and os.path.exists(tmp_ref.name):
                os.unlink(tmp_ref.name)

        buf = io.BytesIO()
        sf.write(buf, wav, self.sample_rate, format="WAV")
        audio = buf.getvalue()
        duration = float(len(wav)) / self.sample_rate
        logger.info("TTS 완료 (%.2fs, %d bytes)", duration, len(audio))
        return {
            "output": {
                "audio_base64": base64.b64encode(audio).decode(),
                "sample_rate": self.sample_rate,
                "duration": duration,
            }
        }


@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def tts(item: Dict[str, Any]) -> Dict[str, Any]:
    body = item.get("input", item)
    try:
        return XTTS().synthesize.remote(body)
    except Exception as e:  # noqa: BLE001
        logger.error("TTS 실패: %s", e)
        return {"output": {"audio_base64": ""}, "error": str(e), "status": "failed"}


@app.local_entrypoint()
def main():
    r = XTTS().synthesize.remote({"text": "안녕하세요, 엑스티티에스 음성 합성 테스트입니다."})
    out = r.get("output", {})
    print(f"len={len(out.get('audio_base64',''))} sr={out.get('sample_rate')} dur={out.get('duration')}")
