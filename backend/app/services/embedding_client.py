import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DIMENSION = 1024


class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = DEFAULT_EMBEDDING_MODEL
    input_type: Optional[str] = "document"
    device: Optional[str] = None
    batch_size: Optional[int] = 32


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model_name: str
    device: str
    batch_size: int


class VLLMEmbeddingClient:
    """Legacy vLLM embedding API client.

    The active RAG path calls MODAL_EMBEDDING_URL through embed_texts(), but
    several legacy services still use this client. Keep the same public API and
    default it to Qwen3-Embedding-0.6B.
    """

    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def generate_embeddings(
        self, texts: List[str], input_type: str = "document", **kwargs
    ) -> EmbeddingResponse:
        try:
            request_data = EmbeddingRequest(
                texts=texts,
                input_type=input_type,
                **kwargs,
            )
            logger.info(
                "Calling embedding API: count=%s input_type=%s model=%s",
                len(texts),
                input_type,
                request_data.model_name,
            )

            response = await self.client.post(
                f"{self.base_url}/embedding/embed",
                json=request_data.model_dump(),
            )
            if response.status_code != 200:
                logger.error(
                    "Embedding API error: %s - %s", response.status_code, response.text
                )
                raise RuntimeError(f"Embedding API error: {response.status_code}")

            return EmbeddingResponse(**response.json())
        except Exception:
            logger.exception("Failed to generate embeddings")
            raise

    async def batch_embedding(
        self, texts: List[str], input_type: str = "document", **kwargs
    ) -> EmbeddingResponse:
        try:
            request_data = EmbeddingRequest(
                texts=texts,
                input_type=input_type,
                **kwargs,
            )
            logger.info(
                "Calling batch embedding API: count=%s input_type=%s model=%s",
                len(texts),
                input_type,
                request_data.model_name,
            )

            response = await self.client.post(
                f"{self.base_url}/embedding/embed/batch",
                json=request_data.model_dump(),
            )
            if response.status_code != 200:
                logger.error(
                    "Batch embedding API error: %s - %s",
                    response.status_code,
                    response.text,
                )
                raise RuntimeError(f"Batch embedding API error: {response.status_code}")

            return EmbeddingResponse(**response.json())
        except Exception:
            logger.exception("Failed to generate batch embeddings")
            raise

    async def get_embedding_info(self) -> Dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/embedding/embed/info")
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> Dict[str, Any]:
        response = await self.client.post(f"{self.base_url}/embedding/embed/health")
        response.raise_for_status()
        return response.json()


_embedding_client = None


def get_embedding_client() -> VLLMEmbeddingClient:
    global _embedding_client

    if _embedding_client is None:
        from app.core.config import settings

        vllm_url = getattr(settings, "VLLM_BASE_URL", "http://localhost:8001")
        _embedding_client = VLLMEmbeddingClient(base_url=vllm_url)

    return _embedding_client


async def generate_embeddings(
    texts: List[str], input_type: str = "document", **kwargs
) -> List[List[float]]:
    async with get_embedding_client() as client:
        response = await client.generate_embeddings(
            texts, input_type=input_type, **kwargs
        )
        return response.embeddings


async def batch_generate_embeddings(
    texts: List[str], input_type: str = "document", **kwargs
) -> List[List[float]]:
    async with get_embedding_client() as client:
        response = await client.batch_embedding(texts, input_type=input_type, **kwargs)
        return response.embeddings


async def embed_texts(
    texts: List[str], input_type: str = "document"
) -> List[List[float]]:
    """Call the Modal Qwen3-Embedding-0.6B endpoint for RAG embeddings.

    input_type must be "document" or "query". The embedding worker applies the
    Qwen3 retrieval instruction only for query embeddings.
    """
    from app.core.config import settings

    if not texts:
        return []

    url = settings.MODAL_EMBEDDING_URL
    if not url:
        raise RuntimeError("MODAL_EMBEDDING_URL is not configured")

    payload = {"texts": texts, "input_type": input_type}
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]
