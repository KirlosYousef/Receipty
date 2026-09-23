import pytest

from app.llm.embeddings import CachingEmbeddings


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(self.calls)), float(len(text))]


def test_same_text_skips_the_provider():
    inner = CountingEmbedder()
    cache = CachingEmbeddings(inner, max_entries=2)
    first = cache.embed("taco")
    first.append(9.0)
    second = cache.embed("taco")
    assert inner.calls == ["taco"]
    assert second == [1.0, 4.0]


def test_full_cache_drops_the_oldest_text():
    inner = CountingEmbedder()
    cache = CachingEmbeddings(inner, max_entries=1)
    cache.embed("a")
    cache.embed("b")
    cache.embed("a")
    assert inner.calls == ["a", "b", "a"]


def test_zero_size_always_calls_the_provider():
    inner = CountingEmbedder()
    cache = CachingEmbeddings(inner, max_entries=0)
    cache.embed("taco")
    cache.embed("taco")
    assert inner.calls == ["taco", "taco"]


def test_provider_failure_is_not_stored():
    class Boom:
        def embed(self, text: str) -> list[float]:
            del text
            raise RuntimeError("upstream down")

    cache = CachingEmbeddings(Boom(), max_entries=2)
    with pytest.raises(RuntimeError):
        cache.embed("taco")
    with pytest.raises(RuntimeError):
        cache.embed("taco")
