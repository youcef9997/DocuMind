
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.chunking import DocumentChunk
from app.embeddings import EmbeddingService


@dataclass(frozen=True)
class SearchResult:
    content: str
    filename: str
    page_number: int | None
    distance: float


class VectorStore:
    """Persist document embeddings and retrieve relevant passages."""

    def __init__(
        self,
        storage_path: str = "data/chroma",
        embedder: EmbeddingService | None = None,
    ):
        Path(storage_path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=storage_path)
        self.collection = self.client.get_or_create_collection(
            name="documind_documents",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder or EmbeddingService()

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0

        batch_size = 100

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            texts = [chunk.content for chunk in batch]
            embeddings = self.embedder.encode(texts)

            metadata = [
                {
                    "filename": chunk.filename,
                    "page_number": (
                        chunk.page_number
                        if chunk.page_number is not None
                        else -1
                    ),
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in batch
            ]

            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=metadata,
            )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        for content, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            page_number = metadata["page_number"]

            matches.append(
                SearchResult(
                    content=content,
                    filename=metadata["filename"],
                    page_number=(
                        page_number if page_number >= 0 else None
                    ),
                    distance=float(distance),
                )
            )

        return matches

    def count(self) -> int:
        return self.collection.count()
