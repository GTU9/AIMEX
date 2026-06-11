"""검증용 pytest 하네스.

AIMEX_TEST 스키마를 대상으로 그룹 A/B를 검증한다.
- DATABASE_URL을 app import 이전에 TEST 스키마로 강제(.env 보다 env var 우선).
- 외부 서비스는 비활성/스텁(S3_ENABLED=false 등).
- TestClient는 컨텍스트매니저 없이 사용 → lifespan(스타트업 백그라운드 작업) 미실행 → 깨끗한 검증.
"""
import os

# --- app import 이전에 반드시 설정 ---
os.environ["DATABASE_URL"] = "mysql+pymysql://root:root@localhost:3306/AIMEX_TEST"
os.environ["S3_ENABLED"] = "false"
os.environ["VLLM_ENABLED"] = "false"
os.environ["AUTO_FINETUNING_ENABLED"] = "false"
os.environ["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "verify-only-secret")

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import init_database


@pytest.fixture(scope="session", autouse=True)
def _prepare_db():
    """TEST 스키마에 테이블 생성(create_all). 마이그레이션이 깨져 있어 create_all 사용."""
    init_database()
    yield


@pytest.fixture()
def client():
    # raise_server_exceptions=False → 500도 응답 객체로 관찰(테스트가 크래시하지 않음)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def auth_token(client):
    """social-login(user_info 주입)으로 외부 OAuth 없이 JWT 발급."""
    r = client.post(
        "/api/v1/auth/social-login",
        json={
            "provider": "google",
            "user_info": {"id": "verify-user-1", "email": "v@test.com", "name": "Verifier"},
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture()
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}
