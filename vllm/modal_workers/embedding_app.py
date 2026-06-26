"""Modal Qwen3-Embedding-0.6B embedding service.

POST /embed
  {"texts": ["...", "..."], "input_type": "document|query"}
returns
  {"embeddings": [[...1024...]], "dimension": 1024, "model_name": "..."}

Qwen3 embedding models are instruction-aware. Documents are embedded as-is,
while query embeddings use the model's built-in "query" prompt.
"""

import modal

app = modal.App("aimex-embedding")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024


def _download_model():
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(MODEL_NAME)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "transformers>=4.51.0",
        "sentence-transformers>=2.7.0",
        "huggingface_hub>=0.23.4",
        "accelerate>=0.30.0",
        "fastapi[standard]",
    )
    .run_function(_download_model)
)

with image.imports():
    import torch
    from sentence_transformers import SentenceTransformer


@app.cls(image=image, gpu="A10G", scaledown_window=300)
class Embedder:
    @modal.enter()
    def load(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device,
            tokenizer_kwargs={"padding_side": "left"},
        )

    @modal.method()
    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        input_type = (input_type or "document").lower()
        encode_kwargs = {
            "batch_size": 16,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        if input_type == "query":
            try:
                vecs = self.model.encode(texts, prompt_name="query", **encode_kwargs)
            except (KeyError, ValueError):
                task = (
                    "Given a web search query, retrieve relevant passages that answer "
                    "the query"
                )
                prompted = [f"Instruct: {task}\nQuery: {text}" for text in texts]
                vecs = self.model.encode(prompted, **encode_kwargs)
        else:
            vecs = self.model.encode(texts, **encode_kwargs)
        return [v.tolist() for v in vecs]


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def embed(item: dict):
    texts = item.get("texts", [])
    input_type = item.get("input_type", "document")
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return {
            "embeddings": [],
            "dimension": EMBEDDING_DIMENSION,
            "model_name": MODEL_NAME,
            "input_type": input_type,
        }
    vecs = Embedder().embed.remote(texts, input_type)
    return {
        "embeddings": vecs,
        "dimension": EMBEDDING_DIMENSION,
        "model_name": MODEL_NAME,
        "input_type": input_type,
    }
