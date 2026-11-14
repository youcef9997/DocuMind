
from dataclasses import dataclass

from app.chunking import chunk_documents
from app.ingestion import ingest_file
from app.vector_store import VectorStore


@dataclass(frozen=True)
class IndexingResult:
    filename: str
    pages: int
    chunks: int


def index_document(
    filename: str,
    data: bytes,
    store: VectorStore,
) -> IndexingResult:
    """Extract, chunk and persist a document."""

    pages = ingest_file(filename, data)
    chunks = chunk_documents(pages)

    stored_count = store.add_chunks(chunks)

    return IndexingResult(
        filename=filename,
        pages=len(pages),
        chunks=stored_count,
    )
