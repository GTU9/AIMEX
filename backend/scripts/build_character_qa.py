"""
캐릭터 대사 리스트 → SFT QA 쌍 생성 (OpenAI).

data/Final_data/{Character}.json 은 캐릭터 대사 문자열 리스트다.
SFT 학습엔 (question, answer) 쌍이 필요하므로, **대사를 answer 로 두고
GPT 가 그 대사가 자연스러운 답변이 되는 사용자 question 을 생성**한다.

실행:
    cd backend
    python scripts/build_character_qa.py Jinx [최대개수]

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
CHUNK = 10

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


def gen_questions(client: OpenAI, answers: list) -> list:
    prompt = (
        f"다음은 캐릭터 '{CHARACTER}'의 대사 목록입니다. "
        f"각 대사가 자연스러운 '답변'이 되도록, 사용자가 했을 법한 '질문'을 한국어로 만들어주세요.\n"
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
    print(f"📖 {CHARACTER}: 대사 {len(lines)}개 로드")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    qa = []
    for i in range(0, len(lines), CHUNK):
        chunk = lines[i : i + CHUNK]
        try:
            questions = gen_questions(client, chunk)
        except Exception as e:
            print(f"  ⚠️ 청크 {i} 실패: {e}")
            continue
        for q, a in zip(questions, chunk):
            if isinstance(q, str) and q.strip():
                qa.append({"question": q.strip(), "answer": a})
        print(f"  ... {len(qa)}개 누적")

    json.dump(qa, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✅ {len(qa)}개 QA 쌍 저장 -> {OUT}")
    if qa:
        print(f"   샘플: Q={qa[0]['question']} / A={qa[0]['answer']}")


if __name__ == "__main__":
    main()
