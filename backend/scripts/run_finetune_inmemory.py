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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
BACKEND = os.path.join(REPO_ROOT, "backend")
MODAL_DIR = os.path.join(REPO_ROOT, "vllm", "modal_workers")
sys.path.insert(0, BACKEND)
sys.path.insert(0, MODAL_DIR)
sys.path.insert(0, HERE)  # gen_persona 등 동일 디렉터리 모듈

JINX_SYSTEM = (
    "당신은 '징크스(Jinx)'입니다. 폭발물과 총을 사랑하는 광기 어린 말괄량이로, "
    "장난기 가득하고 충동적입니다. 항상 들뜨고 도발적인 반말로, 짧고 에너지 넘치게 대답하세요."
)

ALARAK_SYSTEM = (
    "당신은 알라라크입니다. 오만하고 냉소적이며 자신감 넘치는 어조로 말하되, "
    "사용자의 질문과 요구사항에는 빠짐없이 정확하고 직접 답하세요. 캐릭터성은 어휘와 "
    "어조에만 적용하고 사실, 숫자, 문서 및 도구 결과를 왜곡하지 마세요. 모르는 정보는 "
    "지어내지 말고 모른다고 밝히세요. 괄호나 지문, 행동 묘사 없이 대사로만 답하고, "
    "질문과 무관한 전투 명령이나 모욕을 끼워 넣지 마세요."
)

SYSTEM_PRESETS = {"jinx": JINX_SYSTEM, "alarak": ALARAK_SYSTEM}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--influencer", default="jinx-aug")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--system", default=None)
    ap.add_argument("--preset", choices=sorted(SYSTEM_PRESETS), default="jinx")
    ap.add_argument("--url", default=None,
                    help="배포된 Modal finetuning HTTP endpoint URL")
    ap.add_argument("--auto-persona", action="store_true",
                    help="QA 대사에서 캐릭터 페르소나 system_message 자동 생성")
    args = ap.parse_args()
    args.system = args.system or SYSTEM_PRESETS[args.preset]

    # 0) 자동 페르소나 (캐릭터 무관)
    if args.auto_persona:
        from gen_persona import build_persona
        args.system = build_persona(args.qa)
        print(f"🎭 자동 페르소나: {args.system}")

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
    if args.url:
        import httpx

        with httpx.Client(timeout=7200, follow_redirects=True) as client:
            response = client.post(args.url, json={"input": payload})
            response.raise_for_status()
            result = response.json()
    else:
        with fa.app.run():
            result = fa.run_finetuning.remote(payload)

    # 결과 출력 시 토큰 흔적 제거
    safe = {k: v for k, v in (result or {}).items() if k != "hf_token"}
    print("RESULT:", json.dumps(safe, ensure_ascii=False))


if __name__ == "__main__":
    main()
