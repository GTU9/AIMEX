"""게이트 1 — 그룹 A (DB만, 모킹 불필요).

검증 포인트:
- social-login(user_info 주입) → DB User 생성(쓰기) + JWT 발급
- auth/me, users 목록 → 방금 만든 User를 읽어옴(읽기-after-쓰기 왕복)
- teams/analytics/public-mbti → DB 독립 조회

발견된 결함은 strict xfail로 고정(수정되면 XPASS로 알림):
- GET /api/v1/system/logs/  → 500  (system.py:27 current_user.user_id, dict에 .user_id 접근 버그)
- GET /api/v1/notifications → 404  (notifications 라우터가 api.py에 미등록)
"""
import pytest


def test_social_login_creates_user_and_token(client):
    r = client.post(
        "/api/v1/auth/social-login",
        json={"provider": "google",
              "user_info": {"id": "ga-user", "email": "ga@test.com", "name": "GA"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["access_token"]


def test_auth_me_read_after_write(client, auth_headers):
    r = client.get("/api/v1/auth/me", headers=auth_headers)
    assert r.status_code == 200, r.text
    assert r.json()["email"] == "v@test.com"  # conftest auth_token이 만든 유저


def test_users_list(client, auth_headers):
    r = client.get("/api/v1/users", headers=auth_headers)
    assert r.status_code == 200, r.text
    emails = {u.get("email") for u in r.json()}
    assert "v@test.com" in emails


def test_teams_list(client, auth_headers):
    r = client.get("/api/v1/teams", headers=auth_headers)
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list)


def test_public_mbti_no_auth(client):
    r = client.get("/api/v1/public/mbti/")
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list)


def test_analytics_influencer_stats(client, auth_headers):
    r = client.get("/api/v1/analytics/influencers/stats", headers=auth_headers)
    assert r.status_code == 200, r.text
    assert "total_influencers" in r.json()


def test_analytics_board_stats(client, auth_headers):
    r = client.get("/api/v1/analytics/boards/stats", headers=auth_headers)
    assert r.status_code == 200, r.text
    assert "total_boards" in r.json()


@pytest.mark.xfail(strict=True, reason="SECURITY: users.py:73-80 has auth + admin check commented out -> GET /api/v1/users exposes all users(email/id) without a token")
def test_users_list_should_require_auth(client):
    # 토큰 없이 보호되어야 하나, 현재 인증 주석 처리로 200 노출됨
    r = client.get("/api/v1/users")
    assert r.status_code in (401, 403), f"actual={r.status_code} (인증 없이 사용자 목록 노출)"


@pytest.mark.xfail(strict=True, reason="DEFECT: system.py:27 uses current_user.user_id on a dict -> 500 (should be current_user.get('sub'))")
def test_system_logs_should_work(client, auth_headers):
    r = client.get("/api/v1/system/logs/", headers=auth_headers)
    assert r.status_code == 200, f"actual={r.status_code} body={r.text[:120]}"


@pytest.mark.xfail(strict=True, reason="DEFECT: notifications router not registered in api.py -> 404")
def test_notifications_should_be_mounted(client, auth_headers):
    r = client.get("/api/v1/notifications", headers=auth_headers)
    assert r.status_code != 404, f"actual={r.status_code}"
