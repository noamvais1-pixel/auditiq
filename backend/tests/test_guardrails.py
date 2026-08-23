from app.llm.guardrails import REFUSAL_TEXT, verify

ALLOWED = {"vat-basics#1", "depreciation#6"}


def test_valid_citation_passes():
    answer = "שיעור המע\"מ הוא 18% [[vat-basics#1]]."
    cleaned, report = verify(answer, ALLOWED)
    assert report.ok
    assert report.valid_citations == ["vat-basics#1"]
    assert "[[vat-basics#1]]" in cleaned


def test_fabricated_citation_is_stripped_and_reported():
    """The dangerous failure: a well-formed id that was never in the context."""
    answer = "שיעור המע\"מ הוא 18% [[vat-basics#1]] ויש חריג [[vat-basics#99]]."
    cleaned, report = verify(answer, ALLOWED)
    assert report.invalid_citations == ["vat-basics#99"]
    assert "vat-basics#99" not in cleaned
    assert not report.ok


def test_long_uncited_answer_is_downgraded_to_refusal():
    answer = "יש להכיר בהכנסה לפי חמישה שלבים. " * 6
    cleaned, report = verify(answer, ALLOWED)
    assert report.blocked
    assert cleaned == REFUSAL_TEXT


def test_genuine_refusal_is_not_downgraded():
    """A refusal has no citations by construction; blocking it would make the
    correct behaviour unreachable."""
    answer = "לא נמצא במקורות שסופקו מידע על הנושא הזה, ולכן איני עונה. " * 3
    cleaned, report = verify(answer, ALLOWED)
    assert report.is_refusal
    assert not report.blocked
    assert cleaned.startswith("לא נמצא")


def test_short_uncited_answer_is_allowed():
    """Clarifying questions and one-liners must not trip the citation gate."""
    cleaned, report = verify("על איזו שנת מס מדובר?", ALLOWED)
    assert not report.blocked


def test_uncited_claim_is_flagged_not_blocked():
    answer = (
        "שיעור המע\"מ הוא 18% [[vat-basics#1]].\n\n"
        "בנוסף, יש לנכות מס תשומות במלואו בכל מקרה ואין כל מגבלה על כך, "
        "וזה נכון גם לגבי רכב פרטי וגם לגבי הוצאות אירוח בישראל."
    )
    _, report = verify(answer, ALLOWED)
    assert report.uncited_claims
    assert not report.blocked
