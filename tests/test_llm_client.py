from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.llm.client import LLM, LLMError, MissingKey, load_dotenv


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # Caching is on by default in the client, which would make these tests read
    # the developer's real cache directory - and read each *other's* writes,
    # since several of them complete the same prompt "hi". Every test in this
    # file is about the network path, so the cache is off for all of them;
    # caching itself is tested in test_llm_cache.py.
    monkeypatch.setenv("EMAILRAG_LLM_CACHE", "0")


def test_dotenv_is_loaded(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=abc123\n")
    load_dotenv(env)
    assert os.environ["GEMINI_API_KEY"] == "abc123"


def test_dotenv_ignores_comments_blanks_and_quotes(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        'GEMINI_API_KEY="quoted-value"\n'
        "GROQ_API_KEY='single'\n"
        "MALFORMED_NO_EQUALS\n"
        "ANTHROPIC_API_KEY=\n"          # blank value must not be set
    )
    load_dotenv(env)

    assert os.environ["GEMINI_API_KEY"] == "quoted-value"
    assert os.environ["GROQ_API_KEY"] == "single"
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch):
    # A one-off `GEMINI_API_KEY=... make candidates` or a CI secret must
    # override the checked-out file, not the other way round.
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from-file\n")
    load_dotenv(env)

    assert os.environ["GEMINI_API_KEY"] == "from-shell"


def test_missing_dotenv_is_not_an_error(tmp_path):
    load_dotenv(tmp_path / "nope.env")     # must not raise


def test_missing_key_names_the_setup_steps():
    with pytest.raises(MissingKey) as exc:
        LLM("gemini")

    msg = str(exc.value)
    assert "GEMINI_API_KEY" in msg
    assert "aistudio.google.com" in msg
    assert ".env" in msg


def test_an_explicit_api_key_needs_no_env_var():
    # The multi-tenant web app passes each user's own pasted key; it must not
    # need a matching GEMINI_API_KEY in this process's environment at all.
    llm = LLM("gemini", api_key="user-pasted-key")
    assert llm.key == "user-pasted-key"


def test_an_explicit_api_key_wins_over_the_environment(monkeypatch):
    # Explicit always wins: a per-request key must not be silently shadowed by
    # whatever happens to be in this process's own .env/environment.
    monkeypatch.setenv("GEMINI_API_KEY", "process-wide-key")
    llm = LLM("gemini", api_key="user-pasted-key")
    assert llm.key == "user-pasted-key"


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMError, match="unknown provider"):
        LLM("telepathy")


def test_ollama_needs_no_key():
    llm = LLM("ollama")                    # local; must not raise
    assert llm.model == "qwen2.5:3b"


def test_provider_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert LLM("gemini").model == "gemini-2.5-flash"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert LLM("anthropic").model == "claude-haiku-4-5-20251001"


def test_default_model_is_pinned_not_latest(monkeypatch):
    # A "-latest" alias would silently change model between runs, and every
    # table in this repo is meant to be reproducible from the eval set.
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert "latest" not in LLM("gemini").model


def _quota_error(req):
    import io
    import urllib.error
    return urllib.error.HTTPError(
        req.full_url, 429, "Too Many Requests", {},
        io.BytesIO(b'{"error":{"message":"You exceeded your current quota"}}'))


def test_quota_exhaustion_rotates_models_instead_of_retrying(monkeypatch):
    # Gemini meters free-tier quota per model, so an exhausted daily quota is
    # not a transient error: backing off 30s and retrying the same model just
    # fails slower. It must move to the next model, once each.
    from emailrag.llm import client as C

    tried = []

    def fake_urlopen(req, timeout=None):
        tried.append(req.full_url.split("/models/")[1].split(":")[0])
        raise _quota_error(req)

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    with pytest.raises(C.QuotaExhausted) as exc:
        LLM("gemini").complete("hi")

    assert len(tried) == len(set(tried))          # each model tried exactly once
    assert "gemini-2.5-flash" in tried
    assert "gemini-3.5-flash" in tried
    assert "every model tried" in str(exc.value)


def test_a_working_fallback_model_is_used(monkeypatch):
    import io
    from emailrag.llm import client as C

    def fake_urlopen(req, timeout=None):
        model = req.full_url.split("/models/")[1].split(":")[0]
        if model == "gemini-2.5-flash":
            raise _quota_error(req)
        return io.BytesIO(
            b'{"candidates":[{"content":{"parts":[{"text":"recovered"}]}}]}')

    monkeypatch.setattr(C.urllib.request, "urlopen",
                        lambda req, timeout=None: _ctx(fake_urlopen(req, timeout)))
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    llm = LLM("gemini")
    assert llm.complete("hi") == "recovered"
    assert llm.model != "gemini-2.5-flash"        # rotated off the dead model


class _ctx:
    """Minimal context-manager wrapper so a BytesIO works with urlopen()."""
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, *a):
        return False


def test_pinning_a_model_disables_rotation(monkeypatch):
    # An explicit --model is a deliberate choice; silently answering from a
    # different model would make the run unreproducible.
    from emailrag.llm import client as C

    tried = []

    def fake_urlopen(req, timeout=None):
        tried.append(req.full_url.split("/models/")[1].split(":")[0])
        raise _quota_error(req)

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    with pytest.raises(C.QuotaExhausted):
        LLM("gemini", model="gemini-3.5-flash").complete("hi")

    assert tried == ["gemini-3.5-flash"]


def test_a_second_key_is_tried_once_the_first_is_exhausted_on_every_model(monkeypatch):
    # Gemini quota is per key too, not just per model - a key exhausted on
    # every model it was tried under is worth nothing further, but a second
    # key has its own separate quota and deserves the same full model list.
    import io

    from emailrag.llm import client as C

    tried = []

    def fake_urlopen(req, timeout=None):
        key = req.full_url.split("key=")[1]
        model = req.full_url.split("/models/")[1].split(":")[0]
        tried.append((key, model))
        if key == "key-1":
            raise _quota_error(req)
        return _ctx(io.BytesIO(
            b'{"candidates":[{"content":{"parts":[{"text":"recovered"}]}}]}'))

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)

    llm = LLM("gemini", api_key=["key-1", "key-2"])
    assert llm.complete("hi") == "recovered"

    # Every model tried under key-1 before key-2 was ever touched, and the
    # first model tried under key-2 (not wherever key-1 left off) is the one
    # that answered.
    assert all(k == "key-1" for k, _m in tried[:-1])
    assert tried[-1] == ("key-2", "gemini-2.5-flash")


def test_every_key_exhausted_on_every_model_names_the_key_count(monkeypatch):
    from emailrag.llm import client as C

    def fake_urlopen(req, timeout=None):
        raise _quota_error(req)

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(C.QuotaExhausted) as exc:
        LLM("gemini", api_key=["key-1", "key-2"]).complete("hi")

    assert "across all 2 keys" in str(exc.value)


def test_a_falsy_second_key_is_dropped_not_tried(monkeypatch):
    # The web app passes [primary, backup_or_none] straight through - a user
    # with no second key saved should behave exactly like a single-key LLM,
    # not attempt a request with an empty key.
    from emailrag.llm import client as C

    tried = []

    def fake_urlopen(req, timeout=None):
        tried.append(req.full_url.split("key=")[1])
        raise _quota_error(req)

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(C.QuotaExhausted):
        LLM("gemini", api_key=["key-1", None, ""]).complete("hi")

    assert set(tried) == {"key-1"}


def test_reasoning_budget_is_requested(monkeypatch):
    # Reasoning models otherwise spend the output budget on hidden thinking and
    # truncate structured output mid-JSON.
    import io
    import json as _json
    from emailrag.llm import client as C

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured.update(_json.loads(req.data))
        return _ctx(io.BytesIO(
            b'{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}'))

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    LLM("gemini", model="gemini-2.5-flash").complete("hi")
    assert captured["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0


def test_model_rejecting_thinking_budget_is_retried_without_it(monkeypatch):
    # Older models 400 on thinkingConfig. Version-prefix matching would be
    # wrong for "-latest" aliases that move between versions, so the client
    # asks and falls back on rejection.
    import io
    import json as _json
    import urllib.error
    from emailrag.llm import client as C

    seen = []

    def fake_urlopen(req, timeout=None):
        body = _json.loads(req.data)
        seen.append("thinkingConfig" in body["generationConfig"])
        if seen[-1]:
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {},
                io.BytesIO(b'{"error":{"message":"Request contains an invalid argument."}}'))
        return _ctx(io.BytesIO(
            b'{"candidates":[{"content":{"parts":[{"text":"recovered"}]}}]}'))

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    assert LLM("gemini", model="gemini-flash-lite-latest").complete("hi") == "recovered"
    assert seen == [True, False]      # asked, rejected, retried without


def test_empty_response_reports_finish_reason(monkeypatch):
    import io
    from emailrag.llm import client as C

    monkeypatch.setattr(C.urllib.request, "urlopen", lambda req, timeout=None: _ctx(
        io.BytesIO(b'{"candidates":[{"finishReason":"MAX_TOKENS","content":{}}]}')))
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    with pytest.raises(LLMError, match="MAX_TOKENS"):
        LLM("gemini").complete("hi")


def test_json_complete_strips_markdown_fences(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    llm = LLM("gemini")
    # Models wrap JSON in fences regardless of instructions.
    monkeypatch.setattr(llm, "complete",
                        lambda *a, **k: '```json\n{"queries": ["a", "b"]}\n```')

    assert llm.json_complete("p") == {"queries": ["a", "b"]}


def test_json_complete_raises_on_non_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    llm = LLM("gemini")
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "I'm sorry, I can't.")

    with pytest.raises(LLMError, match="did not return JSON"):
        llm.json_complete("p")
