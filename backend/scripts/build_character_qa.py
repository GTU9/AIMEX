"""
캐릭터 대사 리스트 → SFT QA 쌍 생성 (OpenAI). **캐릭터 무관(generic)**.

data/Final_data/{Character}.json 은 캐릭터 대사/지문이 섞인 문자열 리스트다.
SFT 학습엔 (question, answer) 쌍이 필요하므로, **대사를 answer 로 두고
GPT 가 그 대사가 자연스러운 답변이 되는 사용자 question 을 생성**한다.

전처리(지문 제거) — 어떤 캐릭터에도 동작하도록 이름에 의존하지 않는다:
  1) 따옴표가 있는 줄 → 따옴표 안 대사만 추출(무비용)
  2) 짧은 무따옴표 줄(≤SHORT_THRESH) → 대사로 간주해 유지(거대 클린 파일 절약)
  3) 긴 무따옴표 줄 → GPT 로 대사/지문 분리(지문 제거, 인칭 보존)

실행:
    cd backend
    python scripts/build_character_qa.py <Character> [최대개수] [대사당질문수]

    예) python scripts/build_character_qa.py Jinx 999 4
        → 정제된 대사 전량 × 질문 4스타일 = 증강 QA

증강 원리: 같은 대사(answer)에 서로 다른 말투/맥락의 질문을 여러 개 생성한다.
answer 는 정제된 원본 대사 그대로 → 말투 보존, 데이터 양만 확대.

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


# 따옴표(대사) 추출용: 큰따옴표 “ ” " , 작은따옴표 ‘ ’ 안의 텍스트
# 캐릭터 이름에 의존하지 않는다 (어떤 캐릭터/언어에도 동작).
_QUOTE_RE = re.compile(r'[“"]([^“”"]+)[”"]|[‘’]([^‘’]+)[’‘]')
# 무따옴표 줄을 "대사로 그냥 둘지" 판별하는 길이 임계값.
# 이보다 짧은 무따옴표 줄은 대사로 간주(거대 클린 파일 절약), 길면 GPT로 추출.
SHORT_THRESH = 45
# 짧아도 '지문(3인칭 서술)' 신호가 있으면 GPT 로 라우팅 (캐릭터 이름 비의존):
#  - 서술형 과거 종결(~았다/었다/였다/했다/있었다 ...) 로 끝나는 문장
#  - 3인칭 대명사(그는/그가/그녀는 ...) 로 시작/포함
_NARRATION_HINT = re.compile(
    r"(았다|었다|였다|했다|있었다|없었다|되었다|졌다|섰다|왔다|갔다|들렸다|보였다|섰다)[.”\"’']?\s*$"
    r"|(^|\s)(그는|그가|그녀는|그녀가|그들은|그들이)\s"
)


def extract_quoted(line: str) -> list:
    """따옴표 안 발화만 추출(인칭 변환 없음). 따옴표가 없으면 빈 리스트."""
    quotes = [g1 or g2 for g1, g2 in _QUOTE_RE.findall(line)]
    return [q.strip() for q in quotes if q and q.strip()]


def is_narration_suspect(line: str) -> bool:
    """따옴표 없는 줄이 지문일 가능성(길이 또는 서술 신호)."""
    return len(line) > SHORT_THRESH or bool(_NARRATION_HINT.search(line))


# 결정론적 지문 후처리 필터(캐릭터명/어미 열거 비의존):
#  '과거형 평서 종결'(받침 ㅆ + 다, 예: 갔다/했다/맺혔다/떴다/들이밀었다)로 끝나면서
#  대화 신호(문장부호 ?!~… 또는 명시적 1·2인칭 화자)가 없으면 지문으로 본다.
_TRAIL = " .”\"’'"
_PRESENT_TAIL = ("있다", "싶다", "맛있다", "재밌다", "재미있다")  # 현재형 ㅆ받침 예외(평서 과거 아님)
_DIALOGUE_PUNCT = re.compile(r"[?!~…]")
# 이름 앞글자(예: '제이스'의 '제') 오탐을 피하려 다음절 명시형만 사용. 단음절 난/넌은 제외.
_FIRST_SECOND = re.compile(
    r"(?:^|\s)(?:나는|내가|나를|나도|너는|네가|너를|너도|저는|제가|저를|저도|우리는|우린|당신은|당신이)(?:\s|$|[,.])"
)


def _ends_past_declarative(s: str) -> bool:
    """문장이 과거형 평서 종결(받침 ㅆ + '다')로 끝나는가."""
    t = s.rstrip(_TRAIL)
    if len(t) < 2 or not t.endswith("다") or t.endswith(_PRESENT_TAIL):
        return False
    prev = t[-2]
    if not ("가" <= prev <= "힣"):
        return False
    return (ord(prev) - 0xAC00) % 28 == 20  # 종성 ㅆ


def is_pure_narration(s: str) -> bool:
    """추출 후에도 남은 순수 3인칭 서술을 걸러낸다."""
    if not _ends_past_declarative(s):
        return False
    if _DIALOGUE_PUNCT.search(s) or _FIRST_SECOND.search(s):
        return False
    return True


def gpt_extract_dialogue(client: OpenAI, items: list) -> list:
    """긴 무따옴표 항목들에서 '실제 발화 대사'만 추출(지문/행동묘사 제거).
    캐릭터 이름을 쓰지 않으므로 어떤 캐릭터에도 일반화된다.
    반환: 입력과 같은 길이의 리스트(각 원소는 추출 대사 문자열, 대사 없으면 '')."""
    prompt = (
        "다음 항목들은 어떤 캐릭터의 데이터입니다. 각 항목은 '대사(캐릭터가 실제로 한 말)'일 수도, "
        "'지문(상황·행동·배경을 3인칭으로 서술한 문장)'일 수도 있습니다.\n"
        "각 항목에서 캐릭터가 실제로 말한 대사만 남기고, 지문/서술/행동묘사는 제거하세요.\n"
        "- 원문 표현과 인칭을 그대로 유지하세요(1인칭이든 3인칭 자칭이든 바꾸지 마세요).\n"
        "- 한 항목에 대사가 여러 개면 자연스럽게 이어 한 문자열로 합치세요.\n"
        "- 대사가 전혀 없는 순수 지문이면 빈 문자열(\"\")로 두세요.\n"
        "- 입력 순서와 동일한 순서로, 문자열 JSON 배열만 출력하세요(설명 금지).\n"
        f"- 배열 길이는 입력 개수({len(items)})와 정확히 같아야 합니다.\n\n"
        f"{json.dumps(items, ensure_ascii=False)}"
    )
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL if settings.OPENAI_MODEL.startswith("gpt-4o") else "gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 대사/지문 분리 라벨러다. 설명 없이 JSON 배열만 출력한다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    txt = resp.choices[0].message.content.strip()
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    if not m:
        raise ValueError(f"추출 JSON 파싱 실패: {txt[:200]}")
    return json.loads(m.group(0))


def load_lines(client: OpenAI) -> list:
    """원본 → 대사만 정제(캐릭터 무관). 지문 제거 + 중복 제거."""
    data = json.load(open(SRC, encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{SRC} 는 리스트가 아님")
    raw = [s.strip() for s in data if isinstance(s, str) and s.strip()]

    dialogues = []           # 확정 대사
    pending, pending_idx = [], []  # GPT 추출 대상(긴 무따옴표)
    for s in raw:
        q = extract_quoted(s)
        if q:                            # 따옴표 → 안쪽 대사만
            dialogues.extend(q)
        elif is_narration_suspect(s):    # 길거나 서술 신호 → GPT 판별 대기
            pending_idx.append(len(dialogues))
            dialogues.append(None)       # 자리 표시
            pending.append(s)
        else:                            # 짧고 깔끔한 무따옴표 → 대사로 간주
            dialogues.append(s)

    # GPT 배치 추출 (긴 무따옴표만)
    if pending:
        print(f"🤖 긴 무따옴표 {len(pending)}개 GPT 대사 추출 중...")
        extracted = []
        for i in range(0, len(pending), CHUNK):
            chunk = pending[i : i + CHUNK]
            try:
                extracted.extend(gpt_extract_dialogue(client, chunk))
            except Exception as e:
                print(f"  ⚠️ 추출 청크 {i} 실패: {e}")
                extracted.extend([""] * len(chunk))
        for slot, val in zip(pending_idx, extracted):
            dialogues[slot] = (val or "").strip() if isinstance(val, str) else ""

    # None/빈 제거 + 결정론적 지문 후처리 + 중복 제거
    lines, seen, dropped = [], set(), 0
    for d in dialogues:
        if not d:
            continue
        if is_pure_narration(d):
            dropped += 1
            continue
        if d not in seen:
            seen.add(d)
            lines.append(d)
    print(f"🧹 지문 제거: 원본 {len(raw)}개 → 대사 {len(lines)}개 "
          f"(후처리 지문 {dropped}개 추가 제거)")
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

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    lines = load_lines(client)
    print(f"📖 {CHARACTER}: 대사 {len(lines)}개 로드 (질문/대사 = {QUESTIONS_PER_LINE}배 증강)")

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
