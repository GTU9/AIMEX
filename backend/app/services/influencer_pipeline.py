"""인플루언서 생성 파이프라인: 생성 → QA(DB) → Modal 파인튜닝 → adapter_repo(DB).

세 가지를 연결한다:
  1) 인플루언서 생성 파이프라인 (동기 QA 생성)
  2) 데이터베이스 (INFLUENCER_QA_PAIR 저장 + influencer_model_repo 갱신)
  3) Modal (GPU_PROVIDER=modal 파인튜닝 워커)

기존 OpenAI Batch + S3 + RunPod 경로와 독립적인 경량 경로다.
"""
import logging
import os
import time

from app.database import get_db
import app.models  # noqa: F401  (전체 SQLAlchemy 매퍼 등록)
import app.models.conversation  # noqa: F401  (Conversation 매퍼 등록)
from app.models.influencer import AIInfluencer
from app.services.finetuning_service import get_finetuning_service
from app.services.hf_token_resolver import get_token_for_influencer
from app.services.influencers.sync_qa import (
    generate_and_store_qa,
    system_message_for,
)

logger = logging.getLogger(__name__)


async def run_creation_pipeline(influencer_id: str, user_id: str = None) -> bool:
    """생성된 인플루언서에 대해 QA 생성 → Modal 파인튜닝 → DB 반영을 수행."""
    db = next(get_db())
    influencer = None
    try:
        influencer = (
            db.query(AIInfluencer)
            .filter(AIInfluencer.influencer_id == influencer_id)
            .first()
        )
        if not influencer:
            logger.error(f"❌ 파이프라인: 인플루언서 없음 {influencer_id}")
            return False

        logger.info(f"🚀 파이프라인 시작: {influencer.influencer_name} ({influencer_id})")
        influencer.learning_status = 0  # 학습 중
        db.commit()

        # 1) 동기 QA 생성 + DB 저장
        qa = generate_and_store_qa(influencer, db)
        if not qa:
            raise RuntimeError("QA 생성 결과가 비어 있음")

        # 2) HF 토큰/사용자명 (DB 복호화)
        hf_token, hf_username = await get_token_for_influencer(influencer, db)
        if not hf_token:
            raise RuntimeError("HF 토큰 조회 실패 (그룹/인플루언서 토큰 미설정)")

        # 3) HF repo 경로 결정 (기존 값 있으면 재사용)
        ft = get_finetuning_service()
        repo = (influencer.influencer_model_repo or "").strip()
        if not repo:
            english = ft._convert_korean_to_english(influencer.influencer_name)
            repo = f"{hf_username}/aimex-{english}".lower()

        # 4) system_message (인플루언서 system_prompt 우선)
        system_message = system_message_for(influencer)
        epochs = int(os.getenv("TRAINING_EPOCHS", "3"))

        logger.info(f"🎯 파인튜닝 위임: repo={repo}, qa={len(qa)}개, epochs={epochs}")

        # 5) Modal 파인튜닝 (GPU_PROVIDER=modal → adapter_repo 동기 반환)
        adapter_repo = await ft.run_finetuning(
            qa_data=qa,
            system_message=system_message,
            hf_repo_id=repo,
            hf_token=hf_token,
            epochs=epochs,
            task_id=f"ft_{influencer_id}_{int(time.time())}",
            system_prompt=system_message,
            influencer_id=influencer_id,
        )
        if not adapter_repo:
            raise RuntimeError("파인튜닝 실패: adapter_repo 미반환")

        # 6) DB 반영: 챗봇이 사용할 모델 repo + 학습 완료 상태
        influencer.influencer_model_repo = adapter_repo
        influencer.learning_status = 1  # 사용 가능
        db.commit()
        logger.info(
            f"✅ 파이프라인 완료: {influencer.influencer_name} → {adapter_repo} (status=1)"
        )
        return True

    except Exception as e:
        logger.error(f"❌ 파이프라인 실패 ({influencer_id}): {e}", exc_info=True)
        try:
            if influencer is not None:
                influencer.learning_status = 0
                db.commit()
        except Exception:
            db.rollback()
        return False
    finally:
        db.close()
