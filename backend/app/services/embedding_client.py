import httpx
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class EmbeddingRequest(BaseModel):
    """임베딩 요청 모델"""
    texts: List[str]
    model_name: Optional[str] = "BAAI/bge-m3"
    device: Optional[str] = None
    batch_size: Optional[int] = 32

class EmbeddingResponse(BaseModel):
    """임베딩 응답 모델"""
    embeddings: List[List[float]]
    dimension: int
    model_name: str
    device: str
    batch_size: int

class VLLMEmbeddingClient:
    """VLLM 서버의 임베딩 API 클라이언트 (완전한 버전)"""
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def generate_embeddings(self, texts: List[str], **kwargs) -> EmbeddingResponse:
        """텍스트를 임베딩으로 변환"""
        try:
            request_data = EmbeddingRequest(
                texts=texts,
                # device 파라미터 제거 (멀티프로세싱에서 자동으로 GPU 1 사용)
                **kwargs
            )
            
            logger.info(f"🔄 VLLM 임베딩 API 호출: {len(texts)}개 텍스트 (멀티프로세싱 GPU 1)")
            
            response = await self.client.post(
                f"{self.base_url}/embedding/embed",
                json=request_data.dict()
            )
            
            if response.status_code != 200:
                logger.error(f"❌ 임베딩 API 오류: {response.status_code} - {response.text}")
                raise Exception(f"임베딩 API 오류: {response.status_code}")
            
            result = response.json()
            logger.info(f"✅ 임베딩 생성 완료: {len(result['embeddings'])}개")
            
            return EmbeddingResponse(**result)
            
        except Exception as e:
            logger.error(f"❌ 임베딩 생성 실패: {e}")
            raise
    
    async def batch_embedding(self, texts: List[str], **kwargs) -> EmbeddingResponse:
        """배치 임베딩 생성 (대용량 처리용)"""
        try:
            request_data = EmbeddingRequest(
                texts=texts,
                # device 파라미터 제거 (멀티프로세싱에서 자동으로 GPU 1 사용)
                **kwargs
            )
            
            logger.info(f"🔄 VLLM 배치 임베딩 API 호출: {len(texts)}개 텍스트 (멀티프로세싱 GPU 1)")
            
            response = await self.client.post(
                f"{self.base_url}/embedding/embed/batch",
                json=request_data.dict()
            )
            
            if response.status_code != 200:
                logger.error(f"❌ 배치 임베딩 API 오류: {response.status_code} - {response.text}")
                raise Exception(f"배치 임베딩 API 오류: {response.status_code}")
            
            result = response.json()
            logger.info(f"✅ 배치 임베딩 생성 완료: {len(result['embeddings'])}개")
            
            return EmbeddingResponse(**result)
            
        except Exception as e:
            logger.error(f"❌ 배치 임베딩 생성 실패: {e}")
            raise
    
    async def get_embedding_info(self) -> Dict[str, Any]:
        """임베딩 모델 정보 조회"""
        try:
            response = await self.client.get(f"{self.base_url}/embedding/embed/info")
            
            if response.status_code != 200:
                logger.error(f"❌ 임베딩 정보 조회 오류: {response.status_code}")
                raise Exception(f"임베딩 정보 조회 오류: {response.status_code}")
            
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ 임베딩 정보 조회 실패: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """임베딩 모델 상태 확인"""
        try:
            response = await self.client.post(f"{self.base_url}/embedding/embed/health")
            
            if response.status_code != 200:
                logger.error(f"❌ 임베딩 상태 확인 오류: {response.status_code}")
                raise Exception(f"임베딩 상태 확인 오류: {response.status_code}")
            
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ 임베딩 상태 확인 실패: {e}")
            raise

# 전역 클라이언트 인스턴스
_embedding_client = None

def get_embedding_client() -> VLLMEmbeddingClient:
    """전역 임베딩 클라이언트 인스턴스 반환"""
    global _embedding_client
    
    if _embedding_client is None:
        from app.core.config import settings
        # VLLM_BASE_URL 사용 (VLLM 클라이언트와 동일한 설정)
        vllm_url = getattr(settings, 'VLLM_BASE_URL', 'http://localhost:8001')
        _embedding_client = VLLMEmbeddingClient(base_url=vllm_url)
    
    return _embedding_client

async def generate_embeddings(texts: List[str], **kwargs) -> List[List[float]]:
    """VLLM 서버를 통한 임베딩 생성 (간편 함수)"""
    async with get_embedding_client() as client:
        # device 파라미터 제거 (멀티프로세싱에서 자동으로 GPU 1 사용)
        response = await client.generate_embeddings(texts, **kwargs)
        return response.embeddings

async def batch_generate_embeddings(texts: List[str], **kwargs) -> List[List[float]]:
    """VLLM 서버를 통한 배치 임베딩 생성 (간편 함수)"""
    async with get_embedding_client() as client:
        # device 파라미터 제거 (멀티프로세싱에서 자동으로 GPU 1 사용)
        response = await client.batch_embedding(texts, **kwargs)
        return response.embeddings


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Modal bge-m3 임베딩 엔드포인트 호출 (RAG용).

    settings.MODAL_EMBEDDING_URL 로 POST {"texts": [...]} → {"embeddings": [[...]], "dimension": 1024}
    실패 시 예외를 던진다(호출측에서 graceful 폴백 처리).
    """
    from app.core.config import settings

    if not texts:
        return []
    url = settings.MODAL_EMBEDDING_URL
    if not url:
        raise RuntimeError("MODAL_EMBEDDING_URL 미설정")
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json={"texts": texts})
        resp.raise_for_status()
        return resp.json()["embeddings"]