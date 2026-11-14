
from dataclasses import dataclass
from pathlib import Path

import pymupdf


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class DocumentPage:
    filename: str
    page_number: int | None
    content: str


def ingest_file(filename: str, data: bytes) -> list[DocumentPage]:
    """Extract text and source metadata from an uploaded document."""

    if not filename or not isinstance(data, bytes):
        raise ValueError("A valid filename and file bytes are required.")

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if not data:
        raise ValueError("The uploaded file is empty.")

    if len(data) > MAX_FILE_SIZE:
        raise ValueError("The file exceeds the 10 MB limit.")

    pages = []

    if extension == ".pdf":
        try:
            with pymupdf.open(stream=data, filetype="pdf") as document:
                for index, page in enumerate(document):
                    text = page.get_text("text").strip()

                    if text:
                        pages.append(
                            DocumentPage(
                                filename=filename,
                                page_number=index + 1,
                                content=text,
                            )
                        )
        except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
            raise ValueError("Invalid or corrupted PDF.") from exc

    else:
        try:
            text = data.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("The file must contain UTF-8 text.") from exc

        if text:
            pages.append(
                DocumentPage(
                    filename=filename,
                    page_number=None,
                    content=text,
                )
            )

    if not pages:
        raise ValueError("No extractable text found in the document.")

    return pages
