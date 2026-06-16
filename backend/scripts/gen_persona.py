"""정제된 대사(answer)로부터 캐릭터 페르소나 system_message 를 자동 생성(캐릭터 무관).

사용: python scripts/gen_persona.py <char>_qa.json
출력: stdout 에 한 줄 system_message (다른 스크립트에서 캡처해 사용)
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from app.core.config import settings  # noqa: E402
from openai import OpenAI  # noqa: E402


def build_persona(qa_path: str) -> str:
    qa = json.load(open(qa_path, encoding="utf-8"))
    answers = [r["answer"] for r in qa if r.get("answer")][:60]
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    prompt = (
        "다음은 한 캐릭터의 실제 대사 모음입니다. 이 캐릭터를 연기할 챗봇용 "
        "system 프롬프트를 한국어 1~2문장으로 만들어주세요.\n"
        "- 말투/성격/어조의 특징을 담되, 특정 대사를 그대로 베끼지 마세요.\n"
        "- '당신은 ...입니다. 항상 ... 말투로 대답하세요.' 형식.\n"
        "- 지문/행동묘사 없이 '대사로만' 답하라는 지시를 반드시 포함하세요.\n"
        "- 설명 없이 system 프롬프트 문장만 출력하세요.\n\n"
        f"{json.dumps(answers, ensure_ascii=False)}"
    )
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL if settings.OPENAI_MODEL.startswith("gpt-4o") else "gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 캐릭터 페르소나 설계자다. 문장만 출력한다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    return re.sub(r"\s+", " ", resp.choices[0].message.content.strip())


if __name__ == "__main__":
    path = sys.argv[1]
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    print(build_persona(path))
