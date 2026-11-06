import pytest

from tests.answer_helpers import hit


def test_evidence_ids_sources_deduplication_and_snapshot_are_program_controlled():
    from app.rag.evidence import build_evidence

    hits = [hit(), hit(), hit(1, text="The workload uses 100 queries.")]
    bundle = build_evidence("paper-1", hits)
    assert [e.evidence_id for e in bundle.items] == ["E1", "E2"]
    assert [e.chunk_index for e in bundle.items] == [0, 1]
    assert all(e.paper_id == "paper-1" and e.page_number == 10 for e in bundle.items)
    original = bundle.items[0].text
    hits[0].chunk.text = "mutated"
    assert bundle.items[0].text == original
    with pytest.raises(ValueError):
        bundle.items[0].text = "changed"


def test_budget_preserves_whole_chunks_and_reports_omissions():
    from app.rag.evidence import build_evidence

    bundle = build_evidence("paper-1", [hit(i, text="A" * 8) for i in range(3)], max_chars=17, max_items=10)
    assert [e.text for e in bundle.items] == ["A" * 8, "A" * 8]
    assert bundle.omitted_count == 1
    assert len(build_evidence("paper-1", [hit(i) for i in range(3)], max_items=1).items) == 1


@pytest.mark.parametrize("case", ["cross-paper", "duplicate-conflict", "blank-text", "bad-range", "bad-page", "bad-index"])
def test_invalid_evidence_is_rejected(case):
    from app.rag.evidence import build_evidence

    hits = [hit()]
    if case == "cross-paper":
        hits.append(hit(1, paper_id="other"))
    elif case == "duplicate-conflict":
        hits.append(hit(text="Conflicting same source."))
    elif case == "blank-text":
        hits[0].chunk.text = " "
    elif case == "bad-range":
        hits[0].chunk.raw_end_char = 999
    elif case == "bad-page":
        hits[0].chunk.page_number = 0
    elif case == "bad-index":
        hits[0].chunk.chunk_index = -1
    with pytest.raises(ValueError):
        build_evidence("paper-1", hits, max_items=1)


@pytest.mark.parametrize("kwargs", [{"max_chars": 0}, {"max_items": 0}, {"max_chars": True}, {"max_items": 1.5}])
def test_invalid_budget_is_rejected(kwargs):
    from app.rag.evidence import build_evidence
    with pytest.raises(ValueError):
        build_evidence("paper-1", [], **kwargs)


def test_oversized_first_chunk_is_not_silently_truncated():
    from app.rag.evidence import build_evidence
    with pytest.raises(ValueError):
        build_evidence("paper-1", [hit()], max_chars=3)


def test_citation_validation_uses_backend_sources_and_first_use_order():
    from app.rag.evidence import build_evidence, verify_answer

    bundle = build_evidence("paper-1", [hit(), hit(1)])
    result = verify_answer("paper-1", {
        "claims": [{"text": "吞吐量为42。", "evidence_ids": ["E2", "E1"]},
                   {"text": "另一条说明。", "evidence_ids": ["E1"]}],
        "insufficient_evidence": False, "reason": None,
    }, bundle)
    assert result.answer == "吞吐量为42。 [E2][E1]\n另一条说明。 [E1]"
    assert [c.evidence_id for c in result.citations] == ["E2", "E1"]
    assert result.citations[0].raw_text == bundle.items[1].raw_text
    assert result.citations[0].page_number == 10


def test_partial_supported_answer_can_explicitly_report_missing_evidence():
    from app.rag.evidence import build_evidence, verify_answer
    result = verify_answer("paper-1", {
        "claims": [{"text": "已知吞吐量。", "evidence_ids": ["E1"]}],
        "insufficient_evidence": True, "reason": "当前片段未提供实验机器配置。",
    }, build_evidence("paper-1", [hit()]))
    assert result.insufficient_evidence is True
    assert "证据不足" in result.answer
    assert len(result.citations) == 1


@pytest.mark.parametrize("payload", [
    {"claims": [{"text": "x", "evidence_ids": ["E99"]}], "insufficient_evidence": False, "reason": None},
    {"claims": [{"text": "x", "evidence_ids": []}], "insufficient_evidence": False, "reason": None},
    {"claims": [{"text": " ", "evidence_ids": ["E1"]}], "insufficient_evidence": False, "reason": None},
    {"claims": [], "insufficient_evidence": False, "reason": None},
    {"claims": [], "insufficient_evidence": True, "reason": " "},
    {"claims": [], "insufficient_evidence": "true", "reason": "missing"},
    {"claims": [{"text": "x", "evidence_ids": [1]}], "insufficient_evidence": False, "reason": None},
    {"claims": [{"text": "x", "evidence_ids": ["E1", "E1"]}], "insufficient_evidence": False, "reason": None},
    {"claims": [{"text": "x", "evidence_ids": ["E1"], "page_number": 999}], "insufficient_evidence": False, "reason": None},
    {"claims": [], "insufficient_evidence": True, "reason": "missing", "raw_text": "invented"},
    {"claims": [{"text": "x", "evidence_ids": ["E1"]}], "insufficient_evidence": False, "reason": "contradiction"},
])
def test_invalid_answer_or_citation_is_rejected(payload):
    from app.rag.evidence import CitationValidationError, build_evidence, verify_answer
    with pytest.raises(CitationValidationError):
        verify_answer("paper-1", payload, build_evidence("paper-1", [hit()]))


@pytest.mark.parametrize("field,text", [("claim", "结论 [E99]"), ("claim", "结论 [E1]"),
                                       ("reason", "缺少证据 [E999]"), ("claim", "结论 [E1, E99]")])
def test_model_text_cannot_supply_reserved_citation_markers(field, text):
    from app.rag.evidence import CitationValidationError, build_evidence, verify_answer
    payload = {"claims": [{"text": text if field == "claim" else "已知结论。", "evidence_ids": ["E1"]}],
               "insufficient_evidence": field == "reason", "reason": text if field == "reason" else None}
    with pytest.raises(CitationValidationError):
        verify_answer("paper-1", payload, build_evidence("paper-1", [hit()]))
