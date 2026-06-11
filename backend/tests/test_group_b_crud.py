"""게이트 2 — 그룹 B CRUD (DB만).

부팅 블로커가 제거되어 boards 라우터가 정상 import됨을 전제로,
인플루언서/게시글 조회 경로(DB 독립)가 동작하는지 검증한다.
(생성은 FK 시드/소유권/관리자 권한이 필요해 별도 게이트로 분리 — README 참조)
"""


def test_influencers_list_empty(client, auth_headers):
    r = client.get("/api/v1/influencers", headers=auth_headers)
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list)


def test_boards_list_empty(client, auth_headers):
    # boards 라우터가 import된다는 것 자체가 부팅 블로커 해결의 증거
    r = client.get("/api/v1/boards", headers=auth_headers)
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list)


def test_boards_requires_auth(client):
    r = client.get("/api/v1/boards")
    assert r.status_code in (401, 403), r.text
