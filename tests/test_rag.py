
from types import SimpleNamespace

import pytest
from groq import APIConnectionError, APIStatusError
from httpx import Request, Response

from app.rag import RAGService
from app.vector_store import SearchResult


class FakeVectorStore:
    def __init__(self, sources=None):
        self.sources = sources if sources is not None else []

    def search(self, query, top_k=3):
        return self.sources[:top_k]


class FakeChatClient:
    def __init__(self, answer="PostgreSQL is a database. [1]"):
        self.answer = answer
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.answer)
                )
            ]
        )


@pytest.fixture
def sample_source():
    return SearchResult(
        content="PostgreSQL is a relational database.",
        filename="database.pdf",
        page_number=3,
        distance=0.12,
    )


def test_generates_grounded_answer(sample_source):
    client = FakeChatClient()

    service = RAGService(
        FakeVectorStore([sample_source]),
        model="test-model",
        client=client,
    )

    result = service.ask("What is PostgreSQL?")

    assert "PostgreSQL" in result.answer
    assert result.sources == [sample_source]
    assert len(client.calls) == 1


def test_prompt_contains_source_metadata(sample_source):
    client = FakeChatClient()

    service = RAGService(
        FakeVectorStore([sample_source]),
        client=client,
    )

    service.ask("What is PostgreSQL?")

    messages = client.calls[0]["messages"]

    assert "[1] Source: database.pdf, page 3" in (
        messages[1]["content"]
    )
    assert sample_source.content in messages[1]["content"]


def test_no_documents_skips_api_request():
    client = FakeChatClient()

    service = RAGService(
        FakeVectorStore(),
        client=client,
    )

    result = service.ask("What is PostgreSQL?")

    assert result.answer == "No indexed documents are available."
    assert result.sources == []
    assert client.calls == []


def test_empty_question_is_rejected():
    service = RAGService(
        FakeVectorStore(),
        client=FakeChatClient(),
    )

    with pytest.raises(ValueError, match="empty"):
        service.ask("   ")


def test_invalid_top_k_is_rejected():
    service = RAGService(
        FakeVectorStore(),
        client=FakeChatClient(),
    )

    with pytest.raises(ValueError, match="positive"):
        service.ask("Question", top_k=0)


def test_empty_generation_response(sample_source):
    service = RAGService(
        FakeVectorStore([sample_source]),
        client=FakeChatClient(answer=""),
    )

    with pytest.raises(RuntimeError, match="empty answer"):
        service.ask("What is PostgreSQL?")


def test_api_connection_failure(sample_source):
    class FailingClient:
        def create(self, **kwargs):
            raise APIConnectionError(
                request=Request("POST", "https://api.groq.com")
            )

    service = RAGService(
        FakeVectorStore([sample_source]),
        client=FailingClient(),
    )

    with pytest.raises(RuntimeError, match="connect"):
        service.ask("What is PostgreSQL?")


def test_api_status_failure(sample_source):
    class FailingClient:
        def create(self, **kwargs):
            response = Response(
                429,
                request=Request(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                ),
            )
            raise APIStatusError(
                "Rate limit exceeded",
                response=response,
                body=None,
            )

    service = RAGService(
        FakeVectorStore([sample_source]),
        client=FailingClient(),
    )

    with pytest.raises(RuntimeError, match="429"):
        service.ask("What is PostgreSQL?")


def test_generation_settings(sample_source):
    client = FakeChatClient()

    service = RAGService(
        FakeVectorStore([sample_source]),
        model="test-model",
        client=client,
    )

    service.ask("What is PostgreSQL?")

    request = client.calls[0]

    assert request["model"] == "test-model"
    assert request["temperature"] == 0
    assert request["max_completion_tokens"] == 512
