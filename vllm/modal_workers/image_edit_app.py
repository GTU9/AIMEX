"""Modal Serverless GPU Worker - 지시 기반 이미지 수정 (InstructPix2Pix).

이미지 + 텍스트 지시("머리를 파랗게") → 편집된 이미지.
기존 ComfyUI 수정 경로를 대체하는 Modal 경로(기존 코드는 유지).

배포:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal deploy vllm/modal_workers/image_edit_app.py
URL을 backend .env 의 MODAL_IMAGE_EDIT_URL 에 설정.

입출력 계약:
  입력:  {"input": {"image_base64": str, "instruction": str,
                    "num_inference_steps": int=10, "image_guidance_scale": float=1.5,
                    "guidance_scale": float=7.0, "seed": int|null}}
  출력:  {"output": {"image_base64": str(PNG), "width": int, "height": int}}
"""
import base64
import io
from typing import Any, Dict

import modal

app = modal.App("aimex-image-edit")
MODEL = "timbrooks/instruct-pix2pix"


def _download():
    from diffusers import StableDiffusionInstructPix2PixPipeline
    import torch

    StableDiffusionInstructPix2PixPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, safety_checker=None)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "diffusers==0.31.0", "transformers==4.46.3", "accelerate",
        "pillow", "fastapi[standard]",
    )
    .run_function(_download)
)


@app.cls(gpu="A10G", image=image, scaledown_window=120, timeout=600, max_containers=2)
class ImageEditor:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

        self.torch = torch
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            MODEL, torch_dtype=torch.float16, safety_checker=None
        ).to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    @modal.method()
    def edit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from PIL import Image

        img_b64 = payload.get("image_base64")
        instruction = payload.get("instruction")
        if not img_b64 or not instruction:
            raise ValueError("image_base64 와 instruction 은 필수입니다.")

        init = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        # 과도한 해상도 방지(긴 변 768로 제한)
        max_side = 768
        if max(init.size) > max_side:
            r = max_side / max(init.size)
            init = init.resize((int(init.width * r), int(init.height * r)))

        gen = None
        if payload.get("seed") is not None:
            gen = self.torch.Generator(device="cuda").manual_seed(int(payload["seed"]))

        out = self.pipe(
            instruction,
            image=init,
            num_inference_steps=int(payload.get("num_inference_steps", 10)),
            image_guidance_scale=float(payload.get("image_guidance_scale", 1.5)),
            guidance_scale=float(payload.get("guidance_scale", 7.0)),
            generator=gen,
        ).images[0]

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return {
            "output": {
                "image_base64": base64.b64encode(buf.getvalue()).decode(),
                "width": out.width,
                "height": out.height,
            }
        }


@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def edit(item: Dict[str, Any]) -> Dict[str, Any]:
    body = item.get("input", item)
    try:
        return ImageEditor().edit.remote(body)
    except Exception as e:  # noqa: BLE001
        return {"output": {"image_base64": ""}, "error": str(e), "status": "failed"}
