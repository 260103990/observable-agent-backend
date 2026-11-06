from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PdfPage(BaseModel):
    page_number: int
    text: str


class CleanedPage(BaseModel):
    """Original extracted text plus normalized text and transient source mapping.

    Offsets count Python Unicode characters, not bytes or model tokens.
    Each normalized character points to the character that produced it.
    """

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(gt=0)
    raw_text: str
    text: str
    source_offsets: tuple[int, ...] = Field(repr=False)

    @model_validator(mode="after")
    def validate_source_offsets(self) -> Self:
        offsets = self.source_offsets
        if (
            len(offsets) != len(self.text)
            or any(offset < 0 or offset >= len(self.raw_text) for offset in offsets)
            or any(left >= right for left, right in zip(offsets, offsets[1:]))
        ):
            raise ValueError("source_offsets must map every character to ordered raw positions")
        return self


class TextChunk(BaseModel):
    """A single-page chunk with half-open ranges in both text representations.

    chunk_index is zero-based within this page; page_number is PDF one-based.
    The persistence layer will attach the owning paper's identity.
    """

    page_number: int = Field(gt=0)
    chunk_index: int = Field(ge=0)
    text: str
    raw_text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    raw_start_char: int = Field(ge=0)
    raw_end_char: int = Field(gt=0)


class PreparedText(BaseModel):
    pages: list[CleanedPage]
    chunks: list[TextChunk]
