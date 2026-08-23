from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_mode_and_index_size():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["mode"] in {"live", "offline"}
    assert body["chunks_indexed"] > 0
    assert body["audit_chain_valid"] is True


def test_ask_returns_sources_and_an_audit_hash():
    body = client.post("/api/ask", json={"question": "מה שיעור המע\"מ בישראל?"}).json()
    assert body["sources"]
    assert len(body["audit_hash"]) == 64
    assert body["coverage"]["sufficient"] is True


def test_out_of_scope_question_is_refused_without_sources():
    body = client.post("/api/ask", json={"question": "מה מזג האוויר מחר?"}).json()
    assert body["mode"] == "gate"
    assert body["sources"] == []
    assert body["grounding"]["blocked"] is True


def test_redact_preview_reports_findings_without_answering():
    body = client.post(
        "/api/redact/preview", json={"text": "ת.ז 123456782 וטלפון 052-1234567"}
    ).json()
    assert body["summary"] == {"israeli_id": 1, "phone": 1}
    assert "123456782" not in body["redacted_text"]


def test_pii_in_a_question_is_redacted_before_the_model_sees_it():
    body = client.post(
        "/api/ask",
        json={"question": "הלקוח ת.ז 123456782 קנה מבנה. מה שיעור הפחת?"},
    ).json()
    assert body["pii_summary"] == {"israeli_id": 1}


def test_question_too_short_is_rejected():
    assert client.post("/api/ask", json={"question": "?"}).status_code == 422


def test_sources_endpoint_lists_the_corpus():
    docs = client.get("/api/sources").json()["documents"]
    assert len(docs) == 7
    assert all(d["title"] and d["chunks"] > 0 for d in docs)
