"""증강 QA로 Modal 파인튜닝 실행 (토큰은 메모리에서만 처리, 디스크/출력에 남기지 않음).

사용:
    cd <repo root>
    python backend/scripts/run_finetune_inmemory.py \
        --qa backend/scripts/jinx_qa.json \
        --repo GTU9/aimex-lora-jinx-v2 \
        --influencer jinx-aug-370 \
        --epochs 3 \
        --group 1

HF 토큰은 DB에서 복호화해 payload 에만 담아 Modal 로 전달한다.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.path.join(REPO_ROOT, "backend")
MODAL_DIR = os.path.join(REPO_ROOT, "vllm", "modal_workers")
sys.path.insert(0, BACKEND)
sys.path.insert(0, MODAL_DIR)

JINX_SYSTEM = (
    "당신은 '징크스(Jinx)'입니다. 폭발물과 총을 사랑하는 광기 어린 말괄량이로, "
    "장난기 가득하고 충동적입니다. 항상 들뜨고 도발적인 반말로, 짧고 에너지 넘치게 대답하세요."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--influencer", default="jinx-aug")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--system", default=JINX_SYSTEM)
    args = ap.parse_args()

    # 1) DB 에서 HF 토큰 복호화 (메모리에만 유지)
    from app.database import SessionLocal
    import app.models  # noqa: F401  (전체 SQLAlchemy 매퍼 등록)
    import app.models.conversation  # noqa: F401  (Conversation 매퍼 등록)
    from app.services.hf_token_resolver import get_hf_token_resolver

    db = SessionLocal()
    try:
        token, user = get_hf_token_resolver().get_token_by_group(args.group, db)
    finally:
        db.close()
    if not token:
        print("FAIL: HF 토큰 조회 실패")
        sys.exit(1)
    print(f"HF 토큰 확보: user={user} (len={len(token)})")  # 값 비노출

    # 2) QA 로드
    with open(args.qa, encoding="utf-8") as f:
        qa_data = json.load(f)
    print(f"QA 로드: {len(qa_data)}개  / repo={args.repo} / epochs={args.epochs}")

    # 3) Modal 함수 호출 (토큰은 payload 에만)
    import finetuning_app as fa

    payload = {
        "influencer_id": args.influencer,
        "hf_token": token,
        "hf_repo_id": args.repo,
        "base_model": fa.DEFAULT_MODEL,
        "system_message": args.system,
        "qa_data": qa_data,
        "training_epochs": args.epochs,
    }
    with fa.app.run():
        result = fa.run_finetuning.remote(payload)

    # 결과 출력 시 토큰 흔적 제거
    safe = {k: v for k, v in (result or {}).items() if k != "hf_token"}
    print("RESULT:", json.dumps(safe, ensure_ascii=False))


if __name__ == "__main__":
    main()
