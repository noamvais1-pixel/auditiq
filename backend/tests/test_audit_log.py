import json

from app.security import AuditLog


def _log(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl")


def _entry(**kw):
    base = dict(
        event="ask", session_id="s", user="u", input_sha256="a" * 64,
        redacted_prompt="שאלה", pii_summary={}, source_ids=["d#1"],
        model="m", answer_sha256="b" * 64, meta={},
    )
    base.update(kw)
    return base


def test_chain_links_each_record_to_the_previous(tmp_path):
    log = _log(tmp_path)
    first = log.append(**_entry())
    second = log.append(**_entry(redacted_prompt="שאלה שנייה"))
    assert second.prev_hash == first.hash
    assert log.verify() == (True, None)


def test_editing_a_past_record_breaks_the_chain(tmp_path):
    """The whole point: silent edits must be detectable after the fact."""
    log = _log(tmp_path)
    log.append(**_entry())
    log.append(**_entry(redacted_prompt="שנייה"))
    log.append(**_entry(redacted_prompt="שלישית"))

    path = tmp_path / "audit.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["redacted_prompt"] = "טקסט שהוחלף"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    ok, broken_at = log.verify()
    assert not ok
    assert broken_at == rows[1]["ts"]


def test_deleting_a_record_breaks_the_chain(tmp_path):
    log = _log(tmp_path)
    log.append(**_entry())
    log.append(**_entry(redacted_prompt="שנייה"))
    log.append(**_entry(redacted_prompt="שלישית"))

    path = tmp_path / "audit.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([rows[0], rows[2]]) + "\n", encoding="utf-8")

    assert log.verify()[0] is False


def test_raw_question_is_never_written_to_disk(tmp_path):
    """The log proves which question was asked via its hash, without storing it."""
    log = _log(tmp_path)
    log.append(**_entry(redacted_prompt="הלקוח [ת.ז-1] שאל על פחת"))
    contents = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "[ת.ז-1]" in contents
    assert "123456782" not in contents
