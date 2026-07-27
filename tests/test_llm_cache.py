from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.llm import client as C
from emailrag.llm.cache import ResponseCache, cache_key
from emailrag.llm.client import LLM


@pytest.fixture(autouse=True)
def cache_enabled(monkeypatch):
    # The client's own test file disables the cache globally; make sure a stale
    # kill switch from another module cannot silently turn these into no-ops.
    monkeypatch.delenv("EMAILRAG_LLM_CACHE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "x")


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(root=tmp_path / "llm_cache")


class _ctx:
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, *a):
        return False


def _gemini_response(text: str) -> _ctx:
    return _ctx(io.BytesIO(json.dumps(
        {"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode()))


# -- keys -------------------------------------------------------------------

def test_key_is_stable_across_calls(cache):
    a = cache_key("gemini", "m", "prompt", "sys", 0.0, 512)
    b = cache_key("gemini", "m", "prompt", "sys", 0.0, 512)
    assert a == b


@pytest.mark.parametrize("changed", [
    {"provider": "groq"},
    {"model": "other"},
    {"prompt": "different"},
    {"system": "different"},
    {"temperature": 0.7},
    {"max_tokens": 1024},
    {"variant": "1"},
])
def test_every_field_that_can_change_a_response_changes_the_key(changed):
    base = dict(provider="gemini", model="m", prompt="p", system="s",
                temperature=0.0, max_tokens=512, variant="")
    assert cache_key(**base) != cache_key(**{**base, **changed})


def test_model_is_part_of_the_key(cache):
    # The extraction comparison is local-model vs hosted-model. Serving one
    # model's cached answer for the other would fabricate that result.
    cache.put(cache_key("gemini", "flash", "p", "", 0.0, 8), "flash says",
              provider="gemini", model="flash", prompt="p", max_tokens=8)

    assert cache.get(cache_key("gemini", "flash", "p", "", 0.0, 8)) == "flash says"
    assert cache.get(cache_key("gemini", "pro", "p", "", 0.0, 8)) is None


def test_variant_separates_deliberately_repeated_samples(cache):
    # Multi-query expansion asks for k rewrites of one query. Without variant
    # all k samples collapse onto the first cached answer and the expansion
    # silently degrades to a single query.
    for i, text in enumerate(["rewrite a", "rewrite b"]):
        cache.put(cache_key("gemini", "m", "p", "", 0.9, 64, str(i)), text,
                  provider="gemini", model="m", prompt="p", temperature=0.9,
                  max_tokens=64, variant=str(i))

    assert cache.get(cache_key("gemini", "m", "p", "", 0.9, 64, "0")) == "rewrite a"
    assert cache.get(cache_key("gemini", "m", "p", "", 0.9, 64, "1")) == "rewrite b"


# -- storage ----------------------------------------------------------------

def test_roundtrip_and_provenance(cache):
    key = cache_key("gemini", "gemini-2.5-flash", "p", "s", 0.0, 512)
    cache.put(key, "answer", provider="gemini", model="gemini-2.5-flash",
              prompt="p", system="s", max_tokens=512, elapsed_ms=1234.56)

    assert cache.get(key) == "answer"
    record = json.loads(next(cache.root.glob("*/*.json")).read_text())
    # A cache that cannot answer "which model produced this number, and when"
    # is not an experiment log.
    assert record["model"] == "gemini-2.5-flash"
    assert record["prompt"] == "p"
    assert record["cached_utc"].endswith("Z")
    assert record["elapsed_ms"] == 1234.6


def test_miss_returns_none_and_counts(cache):
    assert cache.get(cache_key("gemini", "m", "nothing here", "", 0.0, 8)) is None


def test_truncated_record_is_a_miss_not_a_crash(cache):
    key = cache_key("gemini", "m", "p", "", 0.0, 8)
    cache.put(key, "answer", provider="gemini", model="m", prompt="p", max_tokens=8)
    path = next(cache.root.glob("*/*.json"))
    path.write_text('{"response": "trunc')      # killed mid-write

    assert cache.get(key) is None


def test_disabled_cache_neither_reads_nor_writes(tmp_path, monkeypatch):
    off = ResponseCache(root=tmp_path / "c", enabled=False)
    key = cache_key("gemini", "m", "p", "", 0.0, 8)
    off.put(key, "answer", provider="gemini", model="m", prompt="p", max_tokens=8)

    assert off.get(key) is None
    assert len(off) == 0


def test_env_kill_switch_beats_an_already_constructed_cache(cache, monkeypatch):
    # The process-wide cache is memoised on first use, so the switch has to be
    # read per access - not frozen at construction.
    key = cache_key("gemini", "m", "p", "", 0.0, 8)
    cache.put(key, "answer", provider="gemini", model="m", prompt="p", max_tokens=8)
    assert cache.get(key) == "answer"

    monkeypatch.setenv("EMAILRAG_LLM_CACHE", "0")
    assert cache.active is False
    assert cache.get(key) is None


def test_clear_and_sizes(cache):
    for i in range(3):
        cache.put(cache_key("gemini", "m", f"p{i}", "", 0.0, 8), "a",
                  provider="gemini", model="m", prompt=f"p{i}", max_tokens=8)

    assert len(cache) == 3
    assert cache.size_bytes() > 0
    assert cache.clear() == 3
    assert len(cache) == 0


# -- integration with the client -------------------------------------------

def test_second_identical_call_makes_no_request(cache, monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _gemini_response("cached me")

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    llm = LLM("gemini", cache=cache)

    assert llm.complete("what is due?") == "cached me"
    assert llm.complete("what is due?") == "cached me"
    assert len(calls) == 1                       # the whole point
    assert llm.last_cached is True
    assert cache.stats.hits == 1 and cache.stats.misses == 1


def test_cache_does_not_cross_prompts(cache, monkeypatch):
    monkeypatch.setattr(C.urllib.request, "urlopen",
                        lambda req, timeout=None: _gemini_response(
                            json.loads(req.data)["contents"][0]["parts"][0]["text"].upper()))
    llm = LLM("gemini", cache=cache)

    assert llm.complete("alpha") == "ALPHA"
    assert llm.complete("beta") == "BETA"


def test_a_fallback_model_answer_is_reused_without_re_paying_the_429(cache, monkeypatch):
    # Run one: the default model's daily quota is spent, a fallback answers.
    # Run two must be served from cache with *zero* requests - not one dead
    # request to the exhausted model followed by a hit.
    def fake_urlopen(req, timeout=None):
        model = req.full_url.split("/models/")[1].split(":")[0]
        if model == "gemini-2.5-flash":
            raise urllib_quota_error(req)
        return _gemini_response("from the fallback")

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    assert LLM("gemini", cache=cache).complete("hi") == "from the fallback"

    calls = []
    monkeypatch.setattr(C.urllib.request, "urlopen",
                        lambda req, timeout=None: calls.append(1))
    assert LLM("gemini", cache=cache).complete("hi") == "from the fallback"
    assert calls == []


def test_a_pinned_model_will_not_accept_a_fallbacks_cached_answer(cache, monkeypatch):
    # Rotation-aware lookup must respect the same rule rotation does: an
    # explicit --model is a deliberate choice, so another model's answer is not
    # an acceptable substitute.
    def fake_urlopen(req, timeout=None):
        return _gemini_response("from 3.5")

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    LLM("gemini", model="gemini-3.5-flash", cache=cache).complete("hi")

    calls = []

    def counting(req, timeout=None):
        calls.append(1)
        return _gemini_response("from lite")

    monkeypatch.setattr(C.urllib.request, "urlopen", counting)
    assert LLM("gemini", model="gemini-flash-lite-latest",
               cache=cache).complete("hi") == "from lite"
    assert calls == [1]


def test_failures_are_not_cached(cache, monkeypatch):
    def failing(req, timeout=None):
        raise urllib_error_500(req)

    monkeypatch.setattr(C.urllib.request, "urlopen", failing)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)     # no real backoff
    with pytest.raises(C.LLMError):
        LLM("gemini", cache=cache).complete("hi")

    # A cached error would outlive the transient condition that caused it and
    # be indistinguishable from a real response.
    assert len(cache) == 0


def test_use_cache_false_bypasses_the_shared_cache(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        return _gemini_response("fresh")

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    llm = LLM("gemini", use_cache=False)
    llm.complete("hi")
    llm.complete("hi")

    assert llm.cache is None
    assert len(calls) == 2


def urllib_quota_error(req):
    import urllib.error
    return urllib.error.HTTPError(
        req.full_url, 429, "Too Many Requests", {},
        io.BytesIO(b'{"error":{"message":"You exceeded your current quota"}}'))


def urllib_error_500(req):
    import urllib.error
    return urllib.error.HTTPError(
        req.full_url, 503, "Service Unavailable", {}, io.BytesIO(b"{}"))
