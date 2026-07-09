import time
from app.services.rag_service import TTLCache


def test_cache_set_get():
    cache = TTLCache(ttl=60, maxsize=10)
    cache.set("apa itu gizi?", "jawaban gizi")
    assert cache.get("apa itu gizi?") is not None


def test_cache_miss():
    cache = TTLCache(ttl=60)
    assert cache.get("unknown question") is None


def test_cache_key_normalization():
    cache = TTLCache(ttl=60)
    cache.set("Apa Itu Gizi?", "answer")
    assert cache.get("apa itu gizi?") is not None


def test_cache_ttl_expiry():
    cache = TTLCache(ttl=0)
    cache.set("test", "value")
    assert cache.get("test") is None


def test_cache_maxsize():
    cache = TTLCache(ttl=60, maxsize=2)
    cache.set("q1", "a1")
    cache.set("q2", "a2")
    cache.set("q3", "a3")
    assert cache.get("q1") is None
    assert cache.get("q2") is not None
    assert cache.get("q3") is not None


def test_cache_flush():
    cache = TTLCache(ttl=60)
    cache.set("q1", "a1")
    cache.flush()
    assert cache.get("q1") is None
