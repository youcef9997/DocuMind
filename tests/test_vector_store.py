
import pytest

from app.chunking import DocumentChunk
from app.vector_store import VectorStore


class FakeEmbedder:
    """Provide deterministic vectors without loading an AI model."""

    def encode(self, texts):
        vectors = []

        for text in texts:
            text = text.lower()

            if "python" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "database" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])

        return vectors


@pytest.fixture
def store(tmp_path):
    return VectorStore(
        storage_path=str(tmp_path / "chroma"),
        embedder=FakeEmbedder(),
    )


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="python-001",
            filename="python.md",
            page_number=None,
            content="Python is a programming language.",
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="database-001",
            filename="database.pdf",
            page_number=2,
            content="A database stores structured information.",
            chunk_index=1,
        ),
    ]


def test_add_chunks(store, sample_chunks):
    assert store.add_chunks(sample_chunks) == 2
    assert store.count() == 2


def test_empty_chunks(store):
    assert store.add_chunks([]) == 0
    assert store.count() == 0


def test_upsert_prevents_duplicate_ids(store, sample_chunks):
    store.add_chunks(sample_chunks)
    store.add_chunks(sample_chunks)

    assert store.count() == 2


def test_search_returns_relevant_document(store, sample_chunks):
    store.add_chunks(sample_chunks)

    results = store.search("Explain Python", top_k=1)

    assert len(results) == 1
    assert results[0].filename == "python.md"
    assert "Python" in results[0].content


def test_search_preserves_page_metadata(store, sample_chunks):
    store.add_chunks(sample_chunks)

    results = store.search("database", top_k=1)

    assert results[0].filename == "database.pdf"
    assert results[0].page_number == 2


def test_empty_collection_returns_no_results(store):
    assert store.search("Python") == []


def test_invalid_query(store):
    with pytest.raises(ValueError, match="empty"):
        store.search("   ")


def test_invalid_top_k(store):
    with pytest.raises(ValueError, match="positive"):
        store.search("Python", top_k=0)


def test_persistence_across_instances(
    tmp_path,
    sample_chunks,
):
    path = str(tmp_path / "persistent_db")

    first = VectorStore(path, embedder=FakeEmbedder())
    first.add_chunks(sample_chunks)

    second = VectorStore(path, embedder=FakeEmbedder())

    assert second.count() == 2
    assert second.search("Python", top_k=1)[0].filename == "python.md"
