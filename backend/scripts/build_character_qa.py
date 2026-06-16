"""
캐릭터 대사 리스트 → SFT QA 쌍 생성 (OpenAI).

data/Final_data/{Character}.json 은 캐릭터 대사 문자열 리스트다.
SFT 학습엔 (question, answer) 쌍이 필요하므로, **대사를 answer 로 두고
GPT 가 그 대사가 자연스러운 답변이 되는 사용자 question 을 생성**한다.

실행:
    cd backend
    python scripts/build_character_qa.py Jinx [최대개수] [대사당질문수]

    예) python scripts/build_character_qa.py Jinx 93 4
        → 대사 93개 × 질문 4개 = 최대 372개 QA (증강)

증강 원리: 같은 대사(answer)에 대해 서로 다른 말투/맥락의 질문을 여러 개
생성한다. answer 는 원본 캐릭터 대사 그대로 → 말투 보존, 데이터 양만 확대.

출력: backend/scripts/{character}_qa.json  ([{"question","answer"}, ...])
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from app.core.config import settings

CHARACTER = sys.argv[1] if len(sys.argv) > 1 else "Jinx"
MAX_LINES = int(sys.argv[2]) if len(sys.argv) > 2 else 50
# 대사 1개당 생성할 질문 수 (증강 배수). 1이면 기존 동작.
QUESTIONS_PER_LINE = int(sys.argv[3]) if len(sys.argv) > 3 else 1
CHUNK = 10

# 증강 시 질문 다양성을 위한 스타일 힌트 (패스별로 순환 적용)
STYLE_HINTS = [
    "반말로 친근하게 묻는 질문",
    "정중한 존댓말로 묻는 질문",
    "짧고 직설적인 질문",
    "상황이나 배경을 곁들인 구체적인 질문",
    "호기심 가득한 감탄 섞인 질문",
    "약간 도발적이거나 장난스러운 질문",
]

SRC = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "Final_data", f"{CHARACTER}.json"
)
OUT = os.path.join(os.path.dirname(__file__), f"{CHARACTER.lower()}_qa.json")


def load_lines() -> list:
    data = json.load(open(SRC, encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{SRC} 는 리스트가 아님")
    lines = [s.strip() for s in data if isinstance(s, str) and s.strip()]
    return lines[:MAX_LINES]


def gen_questions(client: OpenAI, answers: list, style: str = "") -> list:
    style_line = f"- 질문 스타일: {style}.\n" if style else ""
    prompt = (
        f"다음은 캐릭터 '{CHARACTER}'의 대사 목록입니다. "
        f"각 대사가 자연스러운 '답변'이 되도록, 사용자가 했을 법한 '질문'을 한국어로 만들어주세요.\n"
        f"{style_line}"
        f"- 입력 순서와 동일한 순서로, 질문 문자열만 JSON 배열로 출력하세요.\n"
        f"- 질문 개수는 입력 대사 개수({len(answers)})와 정확히 같아야 합니다.\n\n"
        f"{json.dumps(answers, ensure_ascii=False)}"
    )
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL if settings.OPENAI_MODEL.startswith("gpt-4o") else "gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 데이터 라벨러다. 설명 없이 JSON 배열만 출력한다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    txt = resp.choices[0].message.content.strip()
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    if not m:
        raise ValueError(f"JSON 배열 파싱 실패: {txt[:200]}")
    return json.loads(m.group(0))


def main():
    if not settings.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY 미설정")
        sys.exit(1)

    lines = load_lines()
    print(f"📖 {CHARACTER}: 대사 {len(lines)}개 로드 (질문/대사 = {QUESTIONS_PER_LINE}배 증강)")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    qa = []
    seen = set()  # (question, answer) 중복 제거
    for p in range(QUESTIONS_PER_LINE):
        style = STYLE_HINTS[p % len(STYLE_HINTS)] if QUESTIONS_PER_LINE > 1 else ""
        print(f"🔁 패스 {p + 1}/{QUESTIONS_PER_LINE}" + (f" — {style}" if style else ""))
        for i in range(0, len(lines), CHUNK):
            chunk = lines[i : i + CHUNK]
            try:
                questions = gen_questions(client, chunk, style)
            except Exception as e:
                print(f"  ⚠️ 청크 {i} 실패: {e}")
                continue
            for q, a in zip(questions, chunk):
                if isinstance(q, str) and q.strip():
                    key = (q.strip(), a)
                    if key in seen:
                        continue
                    seen.add(key)
                    qa.append({"question": q.strip(), "answer": a})
            print(f"  ... {len(qa)}개 누적")

    json.dump(qa, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✅ {len(qa)}개 QA 쌍 저장 -> {OUT}")
    if qa:
        print(f"   샘플: Q={qa[0]['question']} / A={qa[0]['answer']}")


if __name__ == "__main__":
    main()
