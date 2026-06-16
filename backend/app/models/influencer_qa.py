"""인플루언서 QA 쌍 모델.

동기 QA 생성 파이프라인이 만든 (question, answer) 쌍을 DB 에 영속화한다.
(기존 qa_generator 의 save_qa_pairs_to_db 는 임시 파일 stub 이었음 → DB 로 대체)
파인튜닝 단계가 이 테이블에서 QA 를 읽어 Modal 로 학습한다.
"""
import uuid

from sqlalchemy import Column, String, Text, ForeignKey, Index
from sqlalchemy.orm import relationship

from app.models.base import Base, TimestampMixin


class InfluencerQAPair(Base, TimestampMixin):
    """인플루언서 학습용 QA 쌍"""

    __tablename__ = "INFLUENCER_QA_PAIR"

    qa_pair_id = Column(
        String(255),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="QA 쌍 고유 식별자",
    )
    influencer_id = Column(
        String(255),
        ForeignKey("AI_INFLUENCER.influencer_id"),
        nullable=False,
        index=True,
        comment="인플루언서 고유 식별자",
    )
    question = Column(Text, nullable=False, comment="사용자 질문")
    answer = Column(Text, nullable=False, comment="캐릭터 답변(말투 보존)")

    influencer = relationship("AIInfluencer")

    __table_args__ = (
        Index("ix_qa_influencer_id", "influencer_id"),
    )
