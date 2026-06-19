"""
MODEL_MBTI 테이블에 16개 MBTI 유형을 시드한다.
멱등(idempotent): mbti_name 기준으로 이미 존재하는 유형은 건너뛰고, 비어있는 mbti_id에 삽입한다.

실행:
    cd backend && python -m scripts.seed_mbti
"""
import sys
from app.database import SessionLocal
from app.models.influencer import ModelMBTI
# SQLAlchemy 매퍼 등록(관계 해석용) — Conversation 등 의존 모델 로드
import app.models.conversation  # noqa: F401
import app.models.chat_message  # noqa: F401

# (이름, 4축 특성, 말투 설명) — 표준 16유형 순서
MBTI_DATA = [
    ("ISTJ", "내향적, 감각적, 사고적, 판단형", "차분하고 사실 위주의 신중한 말투"),
    ("ISFJ", "내향적, 감각적, 감정적, 판단형", "다정하고 배려하는 따뜻한 말투"),
    ("INFJ", "내향적, 직관적, 감정적, 판단형", "신중하고 통찰력 있는 말투"),
    ("INTJ", "내향적, 직관적, 사고적, 판단형", "논리적이고 간결한 전략가형 말투"),
    ("ISTP", "내향적, 감각적, 사고적, 인식형", "담백하고 실용적인 말투"),
    ("ISFP", "내향적, 감각적, 감정적, 인식형", "부드럽고 감성적인 말투"),
    ("INFP", "내향적, 직관적, 감정적, 인식형", "따뜻하고 이상주의적인 말투"),
    ("INTP", "내향적, 직관적, 사고적, 인식형", "분석적이고 호기심 어린 말투"),
    ("ESTP", "외향적, 감각적, 사고적, 인식형", "활기차고 직설적인 말투"),
    ("ESFP", "외향적, 감각적, 감정적, 인식형", "밝고 유쾌한 말투"),
    ("ENFP", "외향적, 직관적, 감정적, 인식형", "에너지 넘치고 공감하는 말투"),
    ("ENTP", "외향적, 직관적, 사고적, 인식형", "재치있고 도전적인 말투"),
    ("ESTJ", "외향적, 감각적, 사고적, 판단형", "단호하고 체계적인 말투"),
    ("ESFJ", "외향적, 감각적, 감정적, 판단형", "친근하고 배려심 깊은 말투"),
    ("ENFJ", "외향적, 직관적, 감정적, 판단형", "따뜻하고 설득력 있는 말투"),
    ("ENTJ", "외향적, 직관적, 사고적, 판단형", "자신감 있고 주도적인 말투"),
]


def main() -> int:
    db = SessionLocal()
    try:
        existing = db.query(ModelMBTI).all()
        existing_names = {m.mbti_name for m in existing}
        used_ids = {m.mbti_id for m in existing}

        next_id = 1
        inserted = []
        for name, traits, speech in MBTI_DATA:
            if name in existing_names:
                continue
            while next_id in used_ids:
                next_id += 1
            db.add(ModelMBTI(
                mbti_id=next_id,
                mbti_name=name,
                mbti_traits=traits,
                mbti_speech=speech,
            ))
            used_ids.add(next_id)
            inserted.append((next_id, name))

        if inserted:
            db.commit()

        total = db.query(ModelMBTI).count()
        print(f"기존 {len(existing)}개 → 신규 {len(inserted)}개 삽입 → 총 {total}개")
        for mid, name in inserted:
            print(f"  + id={mid} {name}")
        return 0
    except Exception as e:
        db.rollback()
        print(f"시드 실패: {e}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
