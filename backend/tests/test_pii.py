"""The redactor's job is to be trusted, so the tests weight precision heavily."""

import pytest

from app.security.pii import redact

# A valid Israeli ID (passes the check digit) and an invalid one of equal length.
VALID_ID = "123456782"
INVALID_ID = "123456789"


def test_redacts_valid_israeli_id():
    r = redact(f"הלקוח ת.ז {VALID_ID} פנה אלינו")
    assert VALID_ID not in r.text
    assert r.summary() == {"israeli_id": 1}


def test_ignores_nine_digits_that_fail_the_check_digit():
    """A journal reference is not an ID. Over-redacting here is what makes users
    disable the feature entirely."""
    r = redact(f"אסמכתא {INVALID_ID} בספר הקופה")
    assert INVALID_ID in r.text
    assert not r.redacted


def test_invoice_numbers_survive():
    r = redact("חשבונית 400123 על סך 5,000 ש\"ח, אסמכתא 88211")
    assert "400123" in r.text and "88211" in r.text


def test_same_value_gets_one_placeholder():
    """Two mentions of one person must stay linkable, or the model loses the
    thread of who it is reasoning about."""
    r = redact(f"{VALID_ID} הוא בעל המניות. ל-{VALID_ID} גם הלוואה מהחברה.")
    assert r.text.count("[ת.ז-1]") == 2
    assert len(set(f.placeholder for f in r.findings)) == 1


def test_rehydrate_round_trips():
    original = f"ת.ז {VALID_ID}, טלפון 052-1234567, מייל a@b.co.il"
    r = redact(original)
    assert r.rehydrate(r.text) == original


@pytest.mark.parametrize(
    "text,kind",
    [
        ("צור קשר ב-052-1234567", "phone"),
        ("+972-54-9876543", "phone"),
        ("שלח ל-yossi.cohen@bdo.co.il", "email"),
        ("חשבון IL620108000000099999999", "bank_account"),
        ("כרטיס 4580000000000000", "credit_card"),
    ],
)
def test_detects_each_kind(text, kind):
    assert kind in redact(text).summary()


def test_overlapping_matches_claim_once():
    """A card number is also a run of digits; it must not be double-counted."""
    r = redact("כרטיס 4580000000000000 בלבד")
    assert len(r.findings) == 1


def test_empty_and_clean_text():
    assert not redact("").redacted
    assert not redact("שאלה על שיעור הפחת למבנים").redacted
