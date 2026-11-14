
import pytest

from app.chunking import chunk_documents
from app.ingestion import DocumentPage


def make_page(text: str) -> DocumentPage:
    return DocumentPage(
        filename="example.pdf",
        page_number=1,
        content=text,
    )


def test_short_document_produces_one_chunk():
    chunks = chunk_documents([make_page("Hello world")])

    assert len(chunks) == 1
    assert chunks[0].content == "Hello world"
    assert chunks[0].page_number == 1


def test_long_document_produces_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(25))

    chunks = chunk_documents(
        [make_page(text)],
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 3


def test_chunks_preserve_overlap():
    text = "one two three four five six seven eight"

    chunks = chunk_documents(
        [make_page(text)],
        chunk_size=5,
        overlap=2,
    )

    assert chunks[0].content.split()[-2:] == (
        chunks[1].content.split()[:2]
    )


def test_chunk_identifiers_are_deterministic():
    pages = [make_page("A document with important information")]

    first = chunk_documents(pages)
    second = chunk_documents(pages)

    assert first[0].chunk_id == second[0].chunk_id


def test_different_content_produces_different_ids():
    first = chunk_documents([make_page("First version")])
    second = chunk_documents([make_page("Updated version")])

    assert first[0].chunk_id != second[0].chunk_id


def test_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_documents([make_page("Text")], chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError, match="overlap"):
        chunk_documents(
            [make_page("Text")],
            chunk_size=10,
            overlap=10,
        )


def test_empty_pages_produce_no_chunks():
    assert chunk_documents([]) == []


def test_metadata_is_preserved():
    chunks = chunk_documents(
        [make_page("Document content")]
    )

    assert chunks[0].filename == "example.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
