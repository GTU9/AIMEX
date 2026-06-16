"""단일 LoRA 어댑터 추론 검증(캐릭터 무관) + 지문 드리프트 자동 탐지.

토큰은 DB 복호화 후 메모리로만 Modal 생성 엔드포인트에 전달.
사용:
    python scripts/verify_character.py --repo GTU9/aimex-lora-tracer --system "..." --tag tracer
결과: backend/scripts/{tag}_verify.json + 드리프트 요약 stdout
"""
import argparse
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)

GEN_URL = os.environ.get(
    "MODAL_GENERATION_URL",
    "https://sangik0909--aimex-generation-generate.modal.run",
)
PROMPTS = [
    "안녕? 너 누구야?",
    "지금 기분 어때?",
    "취미가 뭐야?",
    "오늘 뭐 하고 싶어?",
    "무서운 거 있어?",
    "심심한데 같이 놀자",
    "너 요즘 어떻게 지내?",
    "나한테 한마디 해줘",
]
# 지문(3인칭 서술/행동묘사) 드리프트 마커
DRIFT_RE = re.compile(
    r"(소리쳤|외쳤|말했다|웃으며|꺼내며|꺼냈다|뛰어|달려|쳐다보|중얼|들며|쥐었다|"
    r"바라보|돌더니|올라와|내쉬|끄덕|는다\.|었다\.|했다\.|있었다)"
)


def get_token():
    from app.database import SessionLocal
    import app.models  # noqa: F401
    import app.models.conversation  # noqa: F401
    from app.services.hf_token_resolver import get_hf_token_resolver

    db = SessionLocal()
    try:
        token, _ = get_hf_token_resolver().get_token_by_group(1, db)
    finally:
        db.close()
    return token


def call(token, repo, system, prompt):
    payload = {"input": {
        "hf_token": token, "hf_repo": repo, "system_message": system,
        "prompt": prompt, "temperature": 0.8, "max_tokens": 256,
    }}
    req = urllib.request.Request(
        GEN_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read().decode("utf-8"))
    out = body.get("output", body)
    return out.get("generated_text", str(out)) if isinstance(out, dict) else str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    token = get_token()
    if not token:
        print("FAIL: no token"); sys.exit(1)

    rows, drift = [], 0
    for p in PROMPTS:
        try:
            resp = call(token, args.repo, args.system, p)
        except Exception as e:
            resp = f"[ERROR] {e}"
        d = bool(DRIFT_RE.search(resp))
        drift += d
        rows.append({"prompt": p, "response": resp, "drift": d})
        print(f"  {'⚠️DRIFT' if d else 'ok    '} | {p[:12]}")

    out = os.path.join(HERE, f"{args.tag}_verify.json")
    json.dump({"repo": args.repo, "system": args.system,
               "drift_count": drift, "total": len(PROMPTS), "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"SAVED {out} | 지문드리프트 {drift}/{len(PROMPTS)}")


if __name__ == "__main__":
    main()
