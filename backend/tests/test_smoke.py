"""게이트 0 — 부팅/스모크."""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "healthy"
    assert body["database"] == "connected"


def test_openapi(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["title"] == "AIMEX API"
