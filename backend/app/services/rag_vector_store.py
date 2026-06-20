"""RAG 벡터 스토어 (Chroma · 임베디드).

문서 청크 임베딩을 인플루언서별로 저장/검색한다.
- 컬렉션: rag_documents (COSINE)
- 메타데이터 influencer_id 로 인플루언서 단위 격리(검색 필터 + 삭제)
- 임베디드(서버리스): 별도 Docker/서버 없이 로컬 파일(uploads/vectors)에 영속화한다.

기존 Milvus 구현과 동일한 공개 API(upsert/search/delete_by_influencer/count_by_influencer)를
유지하므로 호출측 변경이 없다. 임베딩 벡터는 외부(Modal bge-m3, 1024d)에서 받아 그대로 저장한다.
"""

import os
import uuid
import logging
from typing import List, Dict, Optional

import chromadb

from app.core.config import settings

logger = logging.getLogger(__name__)

COLLECTION = "rag_documents"


class RAGVectorStore:
    def __init__(self, path: Optional[str] = None):
        # uploads/vectors 아래에 영속화 (이미지/음성 저장소와 같은 uploads 계열)
        self.path = os.path.abspath(path or settings.VECTOR_DB_PATH)
        self._client = None
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            os.makedirs(self.path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.path)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION,
                metadata={"hnsw:space": "cosine"},  # 코사인 유사도
            )
        return self._collection

    def ensure_collection(self):
        # get_or_create_collection 이 멱등이므로 접근만으로 보장된다.
        _ = self.collection

    def upsert(self, docs: List[Dict]):
        """docs: [{text, embedding, influencer_id, source, chunk_id}]"""
        if not docs:
            return
        ids = [str(uuid.uuid4()) for _ in docs]
        embeddings = [d["embedding"] for d in docs]
        documents = [d["text"][:8000] for d in docs]
        metadatas = [
            {
                "influencer_id": str(d["influencer_id"]),
                "source": str(d.get("source") or ""),
                "chunk_id": int(d["chunk_id"]) if d.get("chunk_id") is not None else 0,
            }
            for d in docs
        ]
        self.collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def search(
        self,
        query_vec: List[float],
        influencer_id: str,
        top_k: int = 4,
        threshold: float = 0.6,
    ) -> List[Dict]:
        # 인플루언서 격리 + 코사인 유사도 검색
        res = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where={"influencer_id": str(influencer_id)},
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: List[Dict] = []
        for text, meta, dist in zip(docs, metas, dists):
            # cosine distance(=1-유사도) → 유사도로 환산
            score = 1.0 - float(dist)
            if score >= threshold:
                meta = meta or {}
                out.append(
                    {
                        "text": text or "",
                        "score": score,
                        "source": meta.get("source"),
                        "chunk_id": meta.get("chunk_id"),
                    }
                )
        return out

    def delete_by_influencer(self, influencer_id: str):
        self.collection.delete(where={"influencer_id": str(influencer_id)})

    def count_by_influencer(self, influencer_id: str) -> int:
        res = self.collection.get(
            where={"influencer_id": str(influencer_id)}, include=[]
        )
        return len(res.get("ids") or [])


_store: Optional[RAGVectorStore] = None


def get_vector_store() -> RAGVectorStore:
    global _store
    if _store is None:
        _store = RAGVectorStore()
    return _store
