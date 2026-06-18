"""Modal Serverless GPU Worker - SDXL-Turbo 텍스트→이미지 생성 (ComfyUI 대체).

빠른 1~4 스텝 이미지 생성(stabilityai/sdxl-turbo, ungated, A10G 적합).
모델 가중치는 빌드 시 이미지에 베이크하여 콜드 스타트를 단축한다.

배포:
    cd d:/study/project/AI/AIMEX && \
      PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
      modal deploy vllm/modal_workers/image_gen_app.py
배포 후 URL을 backend .env 의 MODAL_IMAGE_URL 에 설정.

입출력 계약 (aimex-tts/embedding 워커와 동일한 {"input": ...}/{"output": ...} 형태):
  입력:  {"input": {"prompt": str, "negative_prompt": str|null,
                     "width": int=512, "height": int=512,
                     "seed": int|null, "num_inference_steps": int=2,
                     "guidance_scale": float=0.0}}
  출력:  {"output": {"image_base64": <PNG base64>, "width": int,
                     "height": int, "seed": int}}
  오류:  {"output": {"image_base64": ""}, "error": str, "status": "failed"}
"""
import base64
import io
import logging
import random
from typing import Any, Dict

import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = modal.App("aimex-image")

MODEL_NAME = "stabilityai/sdxl-turbo"


def _download_model():
    from diffusers import AutoPipelineForText2Image

    # 가중치 다운로드(이미지에 베이크) - 콜드 스타트 단축
    AutoPipelineForText2Image.from_pretrained(MODEL_NAME)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "diffusers==0.30.3",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        "safetensors",
        "Pillow",
        "huggingface_hub==0.24.6",
        "fastapi[standard]",
    )
    .run_function(_download_model)
)


@app.cls(gpu="A10G", image=image, scaledown_window=120, timeout=600, max_containers=2)
class SDXLTurbo:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import AutoPipelineForText2Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("SDXL-Turbo 로드 시작 (device=%s)", device)
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
        )
        self.pipe = self.pipe.to(device)
        self.device = device
        logger.info("SDXL-Turbo 로드 완료")

    @modal.method()
    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        prompt = payload.get("prompt")
        if not prompt:
            raise ValueError("prompt 필드는 필수입니다.")

        negative_prompt = payload.get("negative_prompt") or None
        width = int(payload.get("width") or 512)
        height = int(payload.get("height") or 512)
        num_inference_steps = int(payload.get("num_inference_steps") or 2)
        guidance_scale = float(payload.get("guidance_scale") or 0.0)

        seed = payload.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(
            "이미지 생성 (steps=%d, %dx%d, guidance=%.1f, seed=%d)",
            num_inference_steps, width, height, guidance_scale, seed,
        )

        kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        result = self.pipe(**kwargs)
        pil_image = result.images[0]

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        logger.info("이미지 생성 완료 (%d bytes)", len(png_bytes))

        return {
            "output": {
                "image_base64": base64.b64encode(png_bytes).decode(),
                "width": width,
                "height": height,
                "seed": seed,
            }
        }


@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def generate(item: Dict[str, Any]) -> Dict[str, Any]:
    body = item.get("input", item)
    try:
        return SDXLTurbo().generate.remote(body)
    except Exception as e:  # noqa: BLE001
        logger.error("이미지 생성 실패: %s", e)
        return {"output": {"image_base64": ""}, "error": str(e), "status": "failed"}


@app.local_entrypoint()
def main():
    r = SDXLTurbo().generate.remote({"prompt": "a cute cat astronaut, digital art"})
    out = r.get("output", {})
    print(f"len={len(out.get('image_base64',''))} {out.get('width')}x{out.get('height')} seed={out.get('seed')}")
