"""RAG 벡터 스토어 통합 테스트 (실 Milvus 필요).

Milvus(Docker)가 떠 있어야 한다. 임베딩은 모킹 없이 결정적 벡터를 사용.
"""
import pytest
from app.services.rag_vector_store import RAGVectorStore


@pytest.mark.integration
def test_upsert_search_isolation():
    store = RAGVectorStore()
    store.ensure_collection()
    inf = "pytest-influencer-001"
    store.delete_by_influencer(inf)

    # 결정적 벡터: 첫 축만 1
    v = [1.0] + [0.0] * 1023
    store.upsert([
        {"text": "환불은 14일 이내 가능합니다.", "embedding": v,
         "influencer_id": inf, "source": "p.txt", "chunk_id": 0},
    ])

    hits = store.search(v, influencer_id=inf, top_k=3, threshold=0.5)
    assert any("환불" in h["text"] for h in hits)

    # 격리: 다른 인플루언서는 0건
    assert store.search(v, influencer_id="other-xyz", top_k=3, threshold=0.5) == []

    store.delete_by_influencer(inf)
