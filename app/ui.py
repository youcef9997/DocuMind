
from pathlib import Path
import sys

import streamlit as st


# Allow imports when launched with: streamlit run app/ui.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.indexing import index_document
from app.rag import RAGService
from app.vector_store import VectorStore


st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_vector_store():
    """Reuse the embedding model and persistent database."""
    return VectorStore(
        storage_path=str(PROJECT_ROOT / "data" / "chroma")
    )


def get_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    return st.session_state.messages


st.title("📚 DocuMind")
st.caption(
    "Ask questions about your documents using semantic search "
    "and retrieval-augmented generation."
)

store = get_vector_store()
messages = get_chat_history()

with st.sidebar:
    st.header("Document library")

    st.caption(
        "Upload a folder containing PDF, TXT or Markdown files."
    )

    uploaded_files = st.file_uploader(
        "Choose a document folder",
        type=["pdf", "txt", "md"],
        accept_multiple_files="directory",
        max_upload_size=10,
        help="Maximum 10 MB per file.",
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} files selected.")

    if st.button(
        "Index documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        successes = []
        failures = []

        if len(uploaded_files) > 50:
            st.error(
                "Please select no more than 50 files per batch."
            )
        else:
            progress = st.progress(0, text="Indexing documents")

            for index, uploaded_file in enumerate(uploaded_files):
                try:
                    result = index_document(
                        filename=uploaded_file.name,
                        data=uploaded_file.getvalue(),
                        store=store,
                    )
                    successes.append(result)

                except (ValueError, RuntimeError, OSError) as exc:
                    failures.append(
                        f"{uploaded_file.name}: {exc}"
                    )

                progress.progress(
                    (index + 1) / len(uploaded_files),
                    text=(
                        f"Processed {index + 1}"
                        f"/{len(uploaded_files)} files"
                    ),
                )

            if successes:
                total_chunks = sum(
                    result.chunks for result in successes
                )
                st.success(
                    f"Indexed {len(successes)} documents "
                    f"({total_chunks} chunks processed)."
                )

            for failure in failures:
                st.error(failure)

    st.divider()

    st.metric(
        "Indexed chunks",
        store.count(),
    )

    st.caption("Document vectors are stored locally.")

    if st.button(
        "Clear chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "AI generation uses GroqCloud. Retrieved passages "
        "are sent to Groq when you ask a question."
    )


st.subheader("Chat with your documents")

if store.count() == 0:
    st.info(
        "Your document library is empty. "
        "Upload and index a folder to get started."
    )

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View sources"):
                for index, source in enumerate(
                    message["sources"], start=1
                ):
                    location = source.filename

                    if source.page_number is not None:
                        location += f" — page {source.page_number}"

                    st.markdown(f"**[{index}] {location}**")
                    st.text(source.content)
                    st.divider()


question = st.chat_input(
    "Ask a question about your documents...",
    disabled=store.count() == 0,
)

if question:
    messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                rag = RAGService(store)
                result = rag.ask(question)

            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

            else:
                st.markdown(result.answer)

                if result.sources:
                    with st.expander("View sources", expanded=True):
                        for index, source in enumerate(
                            result.sources, start=1
                        ):
                            location = source.filename

                            if source.page_number is not None:
                                location += (
                                    f" — page {source.page_number}"
                                )

                            st.markdown(
                                f"**[{index}] {location}**"
                            )
                            st.text(source.content)
                            st.divider()

                messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                })

## Step 3 — Add tests for the indexing pipeline



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
