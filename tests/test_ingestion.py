
import pytest
import pymupdf

from app.ingestion import (
    DocumentPage,
    MAX_FILE_SIZE,
    ingest_file,
)


def test_txt_extraction():
    result = ingest_file("notes.txt", b"Hello DocuMind")

    assert result == [
        DocumentPage(
            filename="notes.txt",
            page_number=None,
            content="Hello DocuMind",
        )
    ]


def test_markdown_extraction():
    result = ingest_file("readme.md", b"# Project\nRAG application")

    assert len(result) == 1
    assert "RAG application" in result[0].content


def test_pdf_extraction():
    document = pymupdf.open()

    for text in ["First page", "Second page"]:
        page = document.new_page()
        page.insert_text((72, 72), text)

    pdf_bytes = document.tobytes()
    document.close()

    result = ingest_file("example.pdf", pdf_bytes)

    assert len(result) == 2
    assert result[0].page_number == 1
    assert result[1].page_number == 2
    assert "Second page" in result[1].content


def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        ingest_file("script.exe", b"some data")


def test_empty_file():
    with pytest.raises(ValueError, match="empty"):
        ingest_file("empty.txt", b"")


def test_oversized_file():
    data = b"x" * (MAX_FILE_SIZE + 1)

    with pytest.raises(ValueError, match="10 MB"):
        ingest_file("large.txt", data)


def test_invalid_pdf():
    with pytest.raises(ValueError, match="Invalid or corrupted"):
        ingest_file("broken.pdf", b"not a valid pdf")


def test_blank_document():
    with pytest.raises(ValueError, match="No extractable text"):
        ingest_file("blank.txt", b" \n \t ")


def test_non_utf8_text():
    with pytest.raises(ValueError, match="UTF-8"):
        ingest_file("invalid.txt", b"\xff\xfe")
