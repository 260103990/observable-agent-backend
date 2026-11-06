from app.rag.models import TextChunk
from app.rag.storage_models import RetrievedChunk


def hit(index=0, *, paper_id="paper-1", text="The baseline throughput is 42 requests/s."):
    raw = "  " + text + "  "
    return RetrievedChunk(paper_id=paper_id, score=0.7, chunk=TextChunk(
        page_number=10, chunk_index=index, text=text, raw_text=text,
        start_char=0, end_char=len(text), raw_start_char=2, raw_end_char=len(raw) - 2,
    ))
