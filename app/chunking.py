
from dataclasses import dataclass
from hashlib import sha256

from app.ingestion import DocumentPage


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    filename: str
    page_number: int | None
    content: str
    chunk_index: int


def chunk_documents(
    pages: list[DocumentPage],
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[DocumentChunk]:
    """Split document pages into overlapping, traceable word chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1.")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        words = page.content.split()

        for start in range(0, len(words), step):
            selected_words = words[start:start + chunk_size]

            if not selected_words:
                continue

            text = " ".join(selected_words)
            chunk_index = len(chunks)

            identifier = sha256(
                (
                    f"{page.filename}|{page.page_number}|"
                    f"{start}|{text}"
                ).encode("utf-8")
            ).hexdigest()

            chunks.append(
                DocumentChunk(
                    chunk_id=identifier,
                    filename=page.filename,
                    page_number=page.page_number,
                    content=text,
                    chunk_index=chunk_index,
                )
            )

            if start + chunk_size >= len(words):
                break

    return chunks
