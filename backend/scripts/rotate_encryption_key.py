"""암호화 키 교체(로테이션) 마이그레이션.

기존에 하드코딩 기본키로 암호화돼 있던 HF 토큰을 강한 랜덤 키로 재암호화한다.

절차:
1) 옛 기본키/솔트로 모든 HF 토큰을 복호화(메모리에만 유지)
2) 새 랜덤 ENCRYPTION_KEY / ENCRYPTION_SALT 생성
3) 새 키로 재암호화하여 DB 저장
4) 새 키/솔트를 backend/.env 에 기록(없을 때만)

보안: 키/토큰 값은 stdout 에 절대 출력하지 않는다(길이/개수만 보고).
재실행 안전: .env 에 ENCRYPTION_KEY 가 이미 있으면 중단.
"""

import os
import sys
import secrets

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(HERE)
sys.path.insert(0, BACKEND_ROOT)

ENV_PATH = os.path.join(BACKEND_ROOT, ".env")

# 과거 하드코딩 기본값 (이 값으로 암호화된 기존 데이터를 복호화하기 위해서만 사용)
OLD_KEY = "skn-team-default-encryption-key-2024"
OLD_SALT = "skn-team-salt-2024"


def _abort(msg: str) -> None:
    print(f"ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    # 0) 이미 키가 설정돼 있으면(=이미 마이그레이션됨) 중단
    from dotenv import dotenv_values
    existing = dotenv_values(ENV_PATH) if os.path.exists(ENV_PATH) else {}
    if existing.get("ENCRYPTION_KEY"):
        _abort("backend/.env 에 ENCRYPTION_KEY 가 이미 존재합니다. 재마이그레이션 방지로 중단.")

    from app.core.encryption import AESEncryption

    # vLLM 매퍼 등록 (Conversation 관계 해소)
    import app.models  # noqa: F401
    import app.models.conversation  # noqa: F401
    from app.database import SessionLocal
    from app.models.user import HFTokenManage

    old_enc = AESEncryption(password=OLD_KEY, salt=OLD_SALT)

    db = SessionLocal()
    try:
        tokens = db.query(HFTokenManage).all()
        print(f"대상 토큰 수: {len(tokens)}")

        # 1) 복호화 (옛 키)
        plain_map = {}
        for t in tokens:
            try:
                plain = old_enc.decrypt(t.hf_token_value)
            except Exception as e:  # noqa: BLE001
                _abort(f"옛 키로 복호화 실패(이미 교체됐을 수 있음): {t.hf_manage_id} - {e}")
            if not plain:
                _abort(f"복호화 결과가 빈 값: {t.hf_manage_id}")
            plain_map[t.hf_manage_id] = plain
        print(f"복호화 성공: {len(plain_map)}건")

        # 2) 새 키/솔트 생성
        new_key = secrets.token_urlsafe(48)
        new_salt = secrets.token_urlsafe(24)
        new_enc = AESEncryption(password=new_key, salt=new_salt)

        # 3) 재암호화 + 검증 후 저장
        for t in tokens:
            re_enc = new_enc.encrypt(plain_map[t.hf_manage_id])
            # 라운드트립 검증
            assert new_enc.decrypt(re_enc) == plain_map[t.hf_manage_id], "재암호화 라운드트립 실패"
            t.hf_token_value = re_enc
        db.commit()
        print(f"재암호화 저장 완료: {len(tokens)}건")

        # 4) .env 기록 (키 값은 출력하지 않음)
        with open(ENV_PATH, "a", encoding="utf-8") as f:
            f.write("\n# 민감정보 암호화 키 (자동 생성, 절대 커밋 금지)\n")
            f.write(f"ENCRYPTION_KEY={new_key}\n")
            f.write(f"ENCRYPTION_SALT={new_salt}\n")
        print(f"backend/.env 기록 완료 (key len={len(new_key)}, salt len={len(new_salt)})")
        print("OK: 키 교체 완료. 백엔드를 재시작하세요.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
