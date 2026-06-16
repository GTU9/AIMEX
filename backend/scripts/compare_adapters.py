"""두 LoRA 어댑터(베이스라인 50 vs 증강 370)를 동일 프롬프트로 비교 추론.

토큰은 DB에서 메모리로만 읽어 Modal 생성 엔드포인트에 전달한다.
결과는 backend/scripts/adapter_compare.json 으로 저장(콘솔 인코딩 회피).
"""
import json
import os
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.path.join(REPO_ROOT, "backend")
sys.path.insert(0, BACKEND)

GEN_URL = os.environ.get(
    "MODAL_GENERATION_URL",
    "https://sangik0909--aimex-generation-generate.modal.run",
)
SYSTEM = (
    "당신은 '징크스(Jinx)'입니다. 폭발물과 총을 사랑하는 광기 어린 말괄량이로, "
    "장난기 가득하고 충동적입니다. 항상 들뜨고 도발적인 반말로, 짧고 에너지 넘치게 대답하세요."
)
ADAPTERS = {
    "baseline_50": "GTU9/aimex-lora-jinx",
    "augmented_370": "GTU9/aimex-lora-jinx-v2",
}
PROMPTS = [
    "안녕? 너 누구야?",
    "지금 기분 어때?",
    "취미가 뭐야?",
    "오늘 뭐 하고 싶어?",
    "무서운 거 있어?",
    "총 쏘는 거 좋아해?",
    "심심한데 같이 놀자",
    "너 좀 위험해 보여",
]


def get_token():
    from app.database import SessionLocal
    import app.models  # noqa: F401
    import app.models.conversation  # noqa: F401
    from app.services.hf_token_resolver import get_hf_token_resolver

    db = SessionLocal()
    try:
        token, user = get_hf_token_resolver().get_token_by_group(1, db)
    finally:
        db.close()
    return token


def call(token, hf_repo, prompt):
    payload = {
        "input": {
            "hf_token": token,
            "hf_repo": hf_repo,
            "system_message": SYSTEM,
            "prompt": prompt,
            "temperature": 0.8,
            "max_tokens": 256,
        }
    }
    req = urllib.request.Request(
        GEN_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read().decode("utf-8"))
    out = body.get("output", body)
    if isinstance(out, dict):
        return out.get("generated_text", json.dumps(out, ensure_ascii=False))
    return str(out)


def main():
    token = get_token()
    if not token:
        print("FAIL: no token")
        sys.exit(1)
    print(f"token ok (len={len(token)})")

    results = []
    for prompt in PROMPTS:
        row = {"prompt": prompt}
        for name, repo in ADAPTERS.items():
            try:
                row[name] = call(token, repo, prompt)
            except Exception as e:
                row[name] = f"[ERROR] {e}"
            print(f"  [{name}] {prompt[:15]}... done")
        results.append(row)

    out = os.path.join(BACKEND, "scripts", "adapter_compare.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"SAVED: {out}  ({len(results)} prompts)")


if __name__ == "__main__":
    main()
