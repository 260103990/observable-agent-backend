from pydantic import BaseModel


class PaperImportResult(BaseModel):
    paper_id: str
    title: str
    page_count: int
    chunk_count: int
