from app.models.rag import Documents


def test_documents_has_influencer_id():
    assert hasattr(Documents, "influencer_id")
    assert "influencer_id" in Documents.__table__.columns
