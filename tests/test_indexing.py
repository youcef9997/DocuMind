
import pytest

from app.indexing import index_document


class FakeVectorStore:
    def __init__(self):
        self.chunks = []

    def add_chunks(self, chunks):
        self.chunks.extend(chunks)
        return len(chunks)


def test_indexes_text_document():
    store = FakeVectorStore()

    result = index_document(
        "guide.txt",
        b"PostgreSQL supports relational data storage.",
        store,
    )

    assert result.filename == "guide.txt"
    assert result.pages == 1
    assert result.chunks == 1
    assert len(store.chunks) == 1


def test_preserves_document_metadata():
    store = FakeVectorStore()

    index_document(
        "guide.md",
        b"# Guide\nPython is a programming language.",
        store,
    )

    assert store.chunks[0].filename == "guide.md"
    assert store.chunks[0].page_number is None


def test_rejects_unsupported_files():
    store = FakeVectorStore()

    with pytest.raises(ValueError, match="Unsupported"):
        index_document("malware.exe", b"invalid", store)

    assert store.chunks == []


def test_rejects_empty_documents():
    store = FakeVectorStore()

    with pytest.raises(ValueError, match="empty"):
        index_document("empty.txt", b"", store)

    assert store.chunks == []


def test_indexing_propagates_storage_errors():
    class FailingStore:
        def add_chunks(self, chunks):
            raise RuntimeError("Database unavailable")

    with pytest.raises(RuntimeError, match="Database unavailable"):
        index_document(
            "guide.txt",
            b"Important document information.",
            FailingStore(),
        )
