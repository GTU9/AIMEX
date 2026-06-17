"""RAG 벡터 스토어 (Milvus).

문서 청크 임베딩을 인플루언서별로 저장/검색한다.
- 컬렉션: rag_documents (dim 1024, COSINE)
- 스칼라 필드 influencer_id 로 인플루언서 단위 격리(검색 필터 + 삭제)
Milvus 미가동 시 호출측에서 예외를 받아 graceful 폴백 처리한다.
"""

import json
import logging
from typing import List, Dict, Optional

from pymilvus import MilvusClient, DataType

from app.core.config import settings

logger = logging.getLogger(__name__)

COLLECTION = "rag_documents"
DIM = 1024


class RAGVectorStore:
    def __init__(self, uri: Optional[str] = None, token: Optional[str] = None):
        self.uri = uri or settings.MILVUS_URI
        self.token = token if token is not None else settings.MILVUS_TOKEN
        self._client: Optional[MilvusClient] = None

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            kwargs = {"uri": self.uri}
            if self.token:
                kwargs["token"] = self.token
            self._client = MilvusClient(**kwargs)
        return self._client

    def ensure_collection(self):
        if self.client.has_collection(COLLECTION):
            return
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("influencer_id", DataType.VARCHAR, max_length=255)
        schema.add_field("metadata", DataType.VARCHAR, max_length=4096)

        index = self.client.prepare_index_params()
        index.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 1024},
        )
        self.client.create_collection(
            COLLECTION, schema=schema, index_params=index
        )
        logger.info("✅ rag_documents 컬렉션 생성")

    def upsert(self, docs: List[Dict]):
        """docs: [{text, embedding, influencer_id, source, chunk_id}]"""
        if not docs:
            return
        self.ensure_collection()
        rows = [
            {
                "vector": d["embedding"],
                "text": d["text"][:8000],
                "influencer_id": d["influencer_id"],
                "metadata": json.dumps(
                    {"source": d.get("source"), "chunk_id": d.get("chunk_id")},
                    ensure_ascii=False,
                ),
            }
            for d in docs
        ]
        self.client.insert(COLLECTION, rows)

    def search(
        self,
        query_vec: List[float],
        influencer_id: str,
        top_k: int = 4,
        threshold: float = 0.6,
    ) -> List[Dict]:
        self.ensure_collection()
        res = self.client.search(
            COLLECTION,
            data=[query_vec],
            limit=top_k,
            filter=f'influencer_id == "{influencer_id}"',
            output_fields=["text", "metadata"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            consistency_level="Strong",  # 벡터화 직후 즉시 검색 가능하도록
        )
        out: List[Dict] = []
        for hit in res[0]:
            score = hit.get("distance", 0.0)  # COSINE: 클수록 유사
            if score >= threshold:
                meta = json.loads(hit["entity"].get("metadata") or "{}")
                out.append(
                    {
                        "text": hit["entity"].get("text", ""),
                        "score": score,
                        **meta,
                    }
                )
        return out

    def delete_by_influencer(self, influencer_id: str):
        if self.client.has_collection(COLLECTION):
            self.client.delete(
                COLLECTION, filter=f'influencer_id == "{influencer_id}"'
            )

    def count_by_influencer(self, influencer_id: str) -> int:
        if not self.client.has_collection(COLLECTION):
            return 0
        rows = self.client.query(
            COLLECTION,
            filter=f'influencer_id == "{influencer_id}"',
            output_fields=["pk"],
        )
        return len(rows)


_store: Optional[RAGVectorStore] = None


def get_vector_store() -> RAGVectorStore:
    global _store
    if _store is None:
        _store = RAGVectorStore()
    return _store
