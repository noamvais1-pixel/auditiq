import pytest

from app.config import settings
from app.rag import Index, assess_coverage, chunk_document, load_corpus
from app.rag.hebrew import tokenize


@pytest.fixture(scope="module")
def index():
    return Index().build(load_corpus(settings.corpus_dir))


# ----------------------------------------------------------------- Hebrew
def test_clitic_prefix_is_indexed_alongside_surface_form():
    """A query for פחת must reach a document that says הפחת."""
    assert "פחת" in tokenize("הפחת")
    assert "פחת" in tokenize("לפחת")


def test_short_words_are_not_over_stripped():
    """Stripping מ from מס leaves ס, which matches everything."""
    assert "ס" not in tokenize("מס")


def test_final_letter_forms_are_folded():
    assert tokenize("נכסים") == tokenize("נכסימ")


def test_gershayim_variants_both_index():
    toks = tokenize('מע"מ')
    assert 'מע"מ' in toks and "מעמ" in toks


# ---------------------------------------------------------------- chunking
def test_subclauses_are_separate_citable_chunks():
    """Regression: an earlier clause regex required whitespace after the paren,
    which merged 4(א) and 4(ב) into the preceding chunk and made them uncitable."""
    chunks = chunk_document("t", "## כותרת\n4(א). ראשון כאן.\n4(ב). שני כאן.\n5. שלישי כאן.")
    assert [c.clause for c in chunks] == ["4(א)", "4(ב)", "5"]


def test_breadcrumb_is_carried_into_the_citation():
    chunks = chunk_document(
        "t", "# מסמך\n## פרק\n1. תוכן הסעיף.", {"title": "כותרת המסמך"}
    )
    assert "כותרת המסמך" in chunks[0].citation
    assert "סעיף 1" in chunks[0].citation


def test_embed_text_includes_the_heading_path():
    """A bare clause is ambiguous without its section; the embedding needs it."""
    chunks = chunk_document("t", "# מסמך\n## פחת\n1. שיעור זה יחול.", {"title": "מדריך"})
    assert "פחת" in chunks[0].embed_text


# --------------------------------------------------------------- retrieval
def test_identifier_in_query_is_pinned():
    """Semantic neighbours are not near misses when a number is named."""
    from app.rag.chunker import Chunk
    from app.rag.retriever import Index as Idx

    idx = Idx().build([
        Chunk("d", "d#1", "הוראות לפי תקן 27 בעניין חכירות."),
        Chunk("d", "d#2", "הוראות לפי תקן 29 בעניין חכירות."),
    ])
    top = idx.search("מה אומר תקן 27?", k=2)[0]
    assert top.chunk.chunk_id == "d#1"
    assert top.pinned


def test_identifier_match_respects_digit_boundaries():
    from app.rag.retriever import _mentions

    assert _mentions("לפי תקן 2 בלבד", "2")
    assert not _mentions("לפי תקן 27 בלבד", "2")


@pytest.mark.parametrize(
    "question,expected",
    [
        ("מאיזה רגע מתחילים לרשום פחת על נכס?", "depreciation#6"),
        ("מה שיעור המע\"מ בישראל?", "vat-basics#1"),
        ("מהוונים מסים נדחים לערך נוכחי?", "deferred-tax#9"),
    ],
)
def test_known_questions_retrieve_their_gold_chunk(index, question, expected):
    assert expected in [h.chunk.chunk_id for h in index.search(question, k=6)]


# ---------------------------------------------------------------- coverage
def test_out_of_domain_question_fails_the_lexical_gate(index):
    q = "מה מזג האוויר בתל אביב מחר?"
    assert not assess_coverage(q, index, index.search(q, k=6)).sufficient


def test_in_domain_questions_all_pass_the_gate(index):
    """Zero false refusals is the requirement; a gate that rejects real questions
    gets routed around by users and protects nothing."""
    for q in [
        "מאיזה רגע מתחילים לרשום פחת על נכס?",
        "ארחתי לקוח במסעדה, מותר לנכות את המע\"מ?",
        "החברה כמעט באיזון תפעולי. איזה מדד מהותיות לבחור?",
    ]:
        assert assess_coverage(q, index, index.search(q, k=6)).sufficient, q
