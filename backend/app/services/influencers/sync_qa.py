"""빠른 동기 QA 생성 (인플루언서 프로필 기반).

기존 OpenAI Batch(비동기, 수분~수시간) 대신, 인플루언서의 성격/말투/설명/MBTI 로부터
GPT(chat completions)로 즉시 QA 쌍을 생성하고 INFLUENCER_QA_PAIR 테이블에 저장한다.
파인튜닝 단계는 이 테이블에서 QA 를 읽는다(S3 의존 제거).
"""
import json
import logging
import os
import re
import uuid
from typing import Dict, List

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.influencer import AIInfluencer
from app.models.influencer_qa import InfluencerQAPair

logger = logging.getLogger(__name__)

# 한 번의 GPT 호출로 생성할 QA 개수
_CHUNK = 12
# 질문 다양성을 위한 스타일(청크별 순환)
_STYLES = [
    "일상적인 안부나 잡담",
    "취미·관심사에 대한 질문",
    "고민 상담이나 조언 요청",
    "감정이나 기분을 묻는 질문",
    "도발적이거나 장난스러운 질문",
    "캐릭터의 정체성·배경에 대한 질문",
]


def build_persona(influencer: AIInfluencer) -> str:
    """인플루언서 필드로부터 페르소나 설명 텍스트 구성(QA 생성·시스템 메시지용)."""
    parts = [f"이름: {influencer.influencer_name}"]
    if getattr(influencer, "influencer_description", None):
        parts.append(f"설명: {influencer.influencer_description}")
    if getattr(influencer, "influencer_personality", None):
        parts.append(f"성격: {influencer.influencer_personality}")
    if getattr(influencer, "influencer_tone", None):
        parts.append(f"말투: {influencer.influencer_tone}")
    mbti = getattr(influencer, "mbti", None)
    if mbti is not None:
        if getattr(mbti, "mbti_name", None):
            parts.append(f"MBTI: {mbti.mbti_name}")
        if getattr(mbti, "mbti_speech", None):
            parts.append(f"MBTI 말투: {mbti.mbti_speech}")
    return "\n".join(parts)


def system_message_for(influencer: AIInfluencer) -> str:
    """파인튜닝/추론에 쓸 system_message. 인플루언서 system_prompt 우선, 없으면 페르소나로 합성."""
    sp = getattr(influencer, "system_prompt", None)
    if sp and sp.strip():
        return sp.strip()
    return (
        f"당신은 '{influencer.influencer_name}'입니다. "
        f"아래 특징을 일관되게 유지하며, 지문이나 행동묘사 없이 '대사'로만 답하세요.\n"
        + build_persona(influencer)
    )


def _gen_chunk(client, persona: str, name: str, style: str, n: int) -> List[Dict]:
    prompt = (
        f"다음은 캐릭터 '{name}'의 프로필입니다.\n{persona}\n\n"
        f"이 캐릭터와 사용자의 대화 QA 쌍을 {n}개 만들어주세요.\n"
        f"- 질문 주제: {style}.\n"
        f"- question 은 사용자가 캐릭터에게 하는 말, answer 는 캐릭터가 위 성격·말투를 살려 하는 답변.\n"
        f"- answer 에는 지문/행동묘사(예: '~하며 웃었다')를 절대 넣지 말고 대사만.\n"
        f"- 출력은 [{{\"question\":\"...\",\"answer\":\"...\"}}, ...] JSON 배열만(설명 금지).\n"
    )
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL if str(getattr(settings, "OPENAI_MODEL", "")).startswith("gpt-4o") else "gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 캐릭터 대화 데이터 생성기다. 설명 없이 JSON 배열만 출력한다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )
    txt = resp.choices[0].message.content.strip()
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    if not m:
        raise ValueError(f"QA JSON 파싱 실패: {txt[:150]}")
    return json.loads(m.group(0))


def generate_and_store_qa(
    influencer: AIInfluencer, db: Session, count: int = None
) -> List[Dict]:
    """동기 QA 생성 후 INFLUENCER_QA_PAIR 에 저장. 저장된 QA 리스트 반환."""
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 미설정")
    from openai import OpenAI

    count = count or int(os.getenv("SYNC_QA_COUNT", "80"))
    persona = build_persona(influencer)
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    name = influencer.influencer_name

    # 재생성 시 기존 QA 제거(중복 방지)
    db.query(InfluencerQAPair).filter(
        InfluencerQAPair.influencer_id == influencer.influencer_id
    ).delete()

    qa: List[Dict] = []
    seen = set()
    i = 0
    while len(qa) < count:
        style = _STYLES[i % len(_STYLES)]
        i += 1
        try:
            chunk = _gen_chunk(client, persona, name, style, _CHUNK)
        except Exception as e:
            logger.warning(f"⚠️ QA 청크 생성 실패(style={style}): {e}")
            if i > len(_STYLES) * 3:  # 무한루프 방지
                break
            continue
        for item in chunk:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if not q or not a or (q, a) in seen:
                continue
            seen.add((q, a))
            qa.append({"question": q, "answer": a})
            db.add(InfluencerQAPair(
                qa_pair_id=str(uuid.uuid4()),
                influencer_id=influencer.influencer_id,
                question=q,
                answer=a,
            ))
        if i > len(_STYLES) * 4:  # 안전 상한
            break

    db.commit()
    logger.info(f"💾 동기 QA {len(qa)}개 저장 완료 (influencer={influencer.influencer_id})")
    return qa


def load_qa_from_db(influencer_id: str, db: Session) -> List[Dict]:
    """저장된 QA 를 파인튜닝용 리스트로 로드."""
    rows = db.query(InfluencerQAPair).filter(
        InfluencerQAPair.influencer_id == influencer_id
    ).all()
    return [{"question": r.question, "answer": r.answer} for r in rows]
