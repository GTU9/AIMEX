"""
로컬 검증용 더미 인플루언서 시드 스크립트 (멱등).

목적: 외부 서비스(OpenAI/vLLM/RunPod) 없이도 프론트엔드의 인플루언서 목록/상세
탭(콘텐츠/음성/API/MCP)에 접근해 UI 흐름을 체험할 수 있도록 최소 더미 데이터를 만든다.

실행:
    cd backend
    python -m scripts.seed_dummy_influencers

전제:
- DB 테이블 생성 완료(AIMEX_MAIN)
- 테스트 사용자(provider_id='devtest-001')와 admin 팀(group_id=1) 존재
  (없으면 자동으로 찾고, 사용자가 없으면 안내 후 종료)
"""

import sys
from app.database import SessionLocal

# SQLAlchemy 매퍼 관계 해소를 위해 모든 모델을 등록한다.
# (app.models.__init__ 은 conversation/image_generation 등 일부 모델을 import하지 않으므로
#  AIInfluencer 의 'Conversation' 등 관계가 해소되지 않는다 → 명시적으로 추가 import)
import app.models  # noqa: F401
from app.models import (  # noqa: F401
    conversation,
    image_generation,
    content_enhancement,
    prompt_optimization,
    image_storage,
)
from app.models.user import User, Team
from app.models.influencer import ModelMBTI, StylePreset, AIInfluencer

DEVTEST_PROVIDER_ID = "devtest-001"
ADMIN_GROUP_ID = 1

DUMMY_INFLUENCERS = [
    {
        "influencer_name": "테스트 인플루언서 - 지나",
        "influencer_description": "로컬 검증용 더미 인플루언서입니다. 밝고 친근한 20대 여성 콘셉트.",
        "influencer_personality": "활발하고 긍정적이며 호기심이 많음",
        "influencer_tone": "친근한 반말과 이모지를 즐겨 사용",
        "influencer_age_group": 20,
        "image_url": "/placeholder-user.jpg",
    },
    {
        "influencer_name": "테스트 인플루언서 - 카이",
        "influencer_description": "로컬 검증용 더미 인플루언서입니다. 차분하고 전문적인 30대 남성 콘셉트.",
        "influencer_personality": "침착하고 분석적이며 신뢰감을 줌",
        "influencer_tone": "정중한 존댓말, 군더더기 없는 설명체",
        "influencer_age_group": 30,
        "image_url": "/placeholder-user.jpg",
    },
]


def get_or_create_mbti(db):
    mbti = db.query(ModelMBTI).filter(ModelMBTI.mbti_id == 16).first()
    if mbti:
        return mbti
    mbti = ModelMBTI(
        mbti_id=16,
        mbti_name="ENFP",
        mbti_traits="외향적, 직관적, 감정적, 인식형",
        mbti_speech="에너지 넘치고 공감하는 말투",
    )
    db.add(mbti)
    db.flush()
    return mbti


def get_or_create_style_preset(db, mbti):
    name = "더미 스타일 프리셋"
    preset = (
        db.query(StylePreset)
        .filter(StylePreset.style_preset_name == name)
        .first()
    )
    if preset:
        return preset
    preset = StylePreset(
        style_preset_name=name,
        influencer_type=0,
        influencer_gender=1,
        influencer_age_group=20,
        influencer_hairstyle="긴 웨이브 헤어",
        influencer_style="청순하고 발랄한 캐주얼",
        influencer_personality="활발하고 친근함",
        influencer_speech="친근한 반말체",
        mbti_id=mbti.mbti_id,
        system_prompt="당신은 밝고 친근한 AI 인플루언서입니다.",
        influencer_description="로컬 검증용 더미 스타일 프리셋",
    )
    db.add(preset)
    db.flush()
    return preset


def main():
    db = SessionLocal()
    try:
        user = (
            db.query(User)
            .filter(User.provider_id == DEVTEST_PROVIDER_ID)
            .first()
        )
        if not user:
            print(
                f"❌ 테스트 사용자(provider_id={DEVTEST_PROVIDER_ID})가 없습니다. "
                "프론트에서 '테스트 로그인'을 한 번 실행한 뒤 다시 시도하세요."
            )
            sys.exit(1)

        team = db.query(Team).filter(Team.group_id == ADMIN_GROUP_ID).first()
        if not team:
            print(f"❌ admin 팀(group_id={ADMIN_GROUP_ID})이 없습니다.")
            sys.exit(1)

        mbti = get_or_create_mbti(db)
        preset = get_or_create_style_preset(db, mbti)

        created, skipped = 0, 0
        for data in DUMMY_INFLUENCERS:
            exists = (
                db.query(AIInfluencer)
                .filter(AIInfluencer.influencer_name == data["influencer_name"])
                .first()
            )
            if exists:
                skipped += 1
                continue

            influencer = AIInfluencer(
                user_id=user.user_id,
                group_id=team.group_id,
                style_preset_id=preset.style_preset_id,
                mbti_id=mbti.mbti_id,
                influencer_name=data["influencer_name"],
                influencer_description=data["influencer_description"],
                image_url=data["image_url"],
                learning_status=1,  # 1 = 사용가능
                influencer_model_repo="dummy/local-test-model",
                chatbot_option=True,
                voice_option=True,
                image_option=True,
                influencer_personality=data["influencer_personality"],
                influencer_tone=data["influencer_tone"],
                influencer_age_group=data["influencer_age_group"],
                system_prompt="당신은 로컬 검증용 더미 AI 인플루언서입니다.",
            )
            db.add(influencer)
            created += 1

        db.commit()
        total = db.query(AIInfluencer).count()
        print(f"✅ 완료 - 생성: {created}개, 건너뜀(이미 존재): {skipped}개")
        print(f"📊 현재 AI_INFLUENCER 총 {total}개")
    except Exception as e:
        db.rollback()
        print(f"❌ 시드 실패: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
