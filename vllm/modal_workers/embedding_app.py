"""Modal bge-m3 임베딩 서비스.

POST /embed  {"texts": ["...", ...]}  ->  {"embeddings": [[...1024...], ...], "dimension": 1024}

RAG 문서/쿼리 임베딩에 사용. backend 가 httpx 로 호출한다.
"""

import modal

app = modal.App("aimex-embedding")

MODEL_NAME = "BAAI/bge-m3"


def _download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "FlagEmbedding==1.2.10",
        "torch==2.3.1",
        "transformers==4.40.2",
        "huggingface_hub==0.23.4",
        "fastapi[standard]",
    )
    .run_function(_download_model)
)

with image.imports():
    from FlagEmbedding import BGEM3FlagModel


@app.cls(image=image, gpu="A10G", scaledown_window=300)
class Embedder:
    @modal.enter()
    def load(self):
        self.model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=32, max_length=2048)["dense_vecs"]
        return [v.tolist() for v in vecs]


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def embed(item: dict):
    texts = item.get("texts", [])
    if not texts:
        return {"embeddings": [], "dimension": 1024}
    vecs = Embedder().embed.remote(texts)
    return {"embeddings": vecs, "dimension": 1024}
