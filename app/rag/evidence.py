"""Build bounded evidence and validate citation membership, not entailment."""
import re

from pydantic import ValidationError

from app.rag.answer_models import AnswerDraft, EvidenceBundle, EvidenceItem, PaperAnswer, VerifiedClaim
from app.rag.storage_models import RetrievedChunk


class CitationValidationError(Exception):
    """The generated structure or references cannot be accepted."""


_RESERVED_MARKER = re.compile(r"\[\s*E[0-9][^\]\n]*\]")


def build_evidence(
    paper_id: str, hits: list[RetrievedChunk], *, max_chars: int = 12000, max_items: int = 10,
) -> EvidenceBundle:
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("paper_id must not be blank")
    if type(max_chars) is not int or max_chars <= 0 or type(max_items) is not int or max_items <= 0:
        raise ValueError("Evidence budgets must be positive integers")
    if not isinstance(hits, list):
        raise ValueError("hits must be a list")
    unique = {}
    # Validate every candidate, including candidates that exceed the budget.
    for hit in hits:
        if not isinstance(hit, RetrievedChunk) or hit.paper_id != paper_id:
            raise ValueError("All evidence must belong to the requested paper")
        item = EvidenceItem(evidence_id="E1", paper_id=paper_id, **hit.chunk.model_dump())
        key = (item.page_number, item.chunk_index)
        if key in unique and unique[key] != item:
            raise ValueError("Conflicting evidence for the same source chunk")
        unique[key] = item

    items = []
    total = 0
    for item in unique.values():
        if len(items) == max_items or total + len(item.text) > max_chars:
            if not items:
                raise ValueError("The first evidence chunk exceeds the character budget")
            break
        total += len(item.text)
        items.append(item.model_copy(update={"evidence_id": f"E{len(items) + 1}"}))
    return EvidenceBundle(paper_id=paper_id, items=tuple(items), omitted_count=len(unique) - len(items))


def verify_answer(paper_id: str, payload: object, evidence: EvidenceBundle) -> PaperAnswer:
    if evidence.paper_id != paper_id or any(e.paper_id != paper_id for e in evidence.items):
        raise CitationValidationError("Evidence does not belong to the requested paper")
    try:
        draft = AnswerDraft.model_validate(payload)
    except ValidationError as exc:
        raise CitationValidationError("Model returned an invalid answer structure") from exc
    if any(_RESERVED_MARKER.search(claim.text) for claim in draft.claims) or (
        draft.reason is not None and _RESERVED_MARKER.search(draft.reason)
    ):
        raise CitationValidationError("Citation markers must be rendered by the backend")

    known = {item.evidence_id: item for item in evidence.items}
    citations = {}
    claims = []
    lines = []
    for claim in draft.claims:
        for evidence_id in claim.evidence_ids:
            if evidence_id not in known:
                raise CitationValidationError("Model cited an unknown evidence ID")
            citations.setdefault(evidence_id, known[evidence_id])
        claims.append(VerifiedClaim(text=claim.text, evidence_ids=tuple(claim.evidence_ids)))
        lines.append(claim.text + " " + "".join(f"[{item}]" for item in claim.evidence_ids))
    reason = draft.reason.strip() if draft.reason is not None else None
    if draft.insufficient_evidence:
        lines.append("证据不足：" + reason)
    return PaperAnswer(
        paper_id=paper_id, answer="\n".join(lines), claims=tuple(claims), citations=tuple(citations.values()),
        insufficient_evidence=draft.insufficient_evidence, reason=reason,
        omitted_evidence_count=evidence.omitted_count,
    )
