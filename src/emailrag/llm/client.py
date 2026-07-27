"""Provider-agnostic LLM client.

Four stages need an LLM - candidate query drafting, commitment extraction, the
query router, and the generation judge - and they do not all want the same
model. Extraction compares a weak local model against a strong hosted one *on
purpose*; the judge wants the strongest thing available because an
under-powered judge produces a low kappa that says more about the judge than
about the system. So provider is a per-call decision, not a global one.

Defaults:
  candidate drafting / router / judge -> Gemini AI Studio free tier ($0)
  extraction quality ceiling          -> Claude Haiku 4.5 (~$3, batched)
  extraction local arm                -> Ollama, CPU-only on this machine

Requests use urllib rather than each vendor's SDK. Three SDKs would add three
dependency trees to a stack already pinned tightly around torch 2.2.2 on Intel
macOS, and all three APIs are a single POST with a JSON body.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .cache import ResponseCache, cache_key

TIMEOUT = 120


def load_dotenv(path: Path | None = None) -> None:
    """Read `.env` from the repo root into os.environ.

    Keys live in a gitignored project file rather than a shell profile: this
    repo is meant to be published, the key is project-scoped, and a global
    export is easy to forget about and leak later.

    Real environment variables always win, so CI secrets and one-off
    `GEMINI_API_KEY=... make candidates` invocations override the file.
    Hand-rolled rather than pulling in python-dotenv - it is twelve lines and
    the dependency tree here is pinned tightly enough already.
    """
    path = path or Path(__file__).resolve().parents[3] / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_dotenv()


class LLMError(RuntimeError):
    pass


class MissingKey(LLMError):
    """Raised with the exact setup step, not a stack trace."""


@dataclass(slots=True)
class Provider:
    name: str
    env_var: str
    model: str
    signup: str


# Gemini free-tier quota is metered PER MODEL, so a 429 on one model says
# nothing about the others - `gemini-2.0-flash` and `-2.0-flash-lite` were
# already exhausted on this key while 2.5/3.5 answered fine. If the default
# throttles, pass --model rather than assuming the key is spent:
#   gemini-2.5-flash  gemini-3.5-flash  gemini-flash-lite-latest
#
# Pinned version, deliberately. `gemini-flash-latest` also works but would
# silently change model between runs, and every table here is supposed to be
# reproducible from the published eval set.
GEMINI_FALLBACKS = ("gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-lite-latest")

PROVIDERS = {
    "gemini": Provider("gemini", "GEMINI_API_KEY", "gemini-2.5-flash",
                       "https://aistudio.google.com/apikey (free, no card)"),
    "groq": Provider("groq", "GROQ_API_KEY", "llama-3.3-70b-versatile",
                     "https://console.groq.com/keys (free)"),
    "anthropic": Provider("anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001",
                          "https://console.anthropic.com/settings/keys"),
    "ollama": Provider("ollama", "", "qwen2.5:3b",
                       "brew install ollama && ollama pull qwen2.5:3b"),
}

DEFAULT = "gemini"

# One cache per process, shared by every LLM instance, so a run that builds
# several clients (the router and the judge, say) reports a single hit rate
# instead of several unrelated ones.
_SHARED_CACHE: ResponseCache | None = None


def default_cache() -> ResponseCache:
    global _SHARED_CACHE
    if _SHARED_CACHE is None:
        _SHARED_CACHE = ResponseCache()
    return _SHARED_CACHE


def _post(url: str, payload: dict, headers: dict, retries: int = 4,
          model_hint: str = "") -> dict:
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:400]
            # A 429 is either a per-minute rate limit (retrying helps) or an
            # exhausted daily quota (retrying just burns 30s to fail anyway).
            # The message distinguishes them, so only back off for the former.
            if "exceeded your current quota" in detail:
                raise QuotaExhausted(
                    f"free-tier quota exhausted for model {model_hint!r}"
                ) from exc
            if exc.code == 429 or exc.code >= 500:
                last = LLMError(f"HTTP {exc.code}: {detail}")
                time.sleep(2 ** attempt * 2)
                continue
            raise LLMError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = LLMError(f"network: {exc}")
            time.sleep(2 ** attempt)
    raise last or LLMError("exhausted retries")


class QuotaExhausted(LLMError):
    """Daily free-tier quota for one specific model is spent."""


class LLM:
    def __init__(self, provider: str = DEFAULT, model: str | None = None,
                 temperature: float = 0.0, auto_fallback: bool = True,
                 cache: ResponseCache | None = None, use_cache: bool = True) -> None:
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider {provider!r}; have {sorted(PROVIDERS)}")
        self.spec = PROVIDERS[provider]
        self.model = model or self.spec.model
        self.temperature = temperature
        # Caching is on by default. The failure mode of forgetting to enable it
        # is a spent daily quota and a `make bench` that cannot finish; the
        # failure mode of forgetting to disable it is a fast, honest re-run.
        # See llm/cache.py. `EMAILRAG_LLM_CACHE=0` overrides this per-run.
        # `cache is not None`, not `cache or ...`: ResponseCache defines
        # __len__, so an empty cache is falsy and `or` would silently swap a
        # caller's explicit cache for the process-wide one.
        self.cache = (cache if cache is not None else default_cache()) if use_cache else None
        # Which model actually produced the last response - a fallback or a
        # cache hit means it is not necessarily `self.model`.
        self.last_model = self.model
        self.last_cached = False
        # Gemini meters quota per model, so one exhausted model says nothing
        # about the rest. Rotating automatically is the difference between a
        # labelling session that finishes and one that dies a third of the way
        # in. Only rotate when the caller did not pin a model explicitly.
        self._fallbacks = list(GEMINI_FALLBACKS) if (
            auto_fallback and provider == "gemini" and model is None) else []
        self._exhausted: set[str] = set()
        self.key = os.environ.get(self.spec.env_var, "") if self.spec.env_var else ""
        if self.spec.env_var and not self.key:
            raise MissingKey(
                f"{self.spec.env_var} is not set.\n"
                f"  1. get a key: {self.spec.signup}\n"
                f"  2. cp .env.example .env\n"
                f"  3. put it in .env as {self.spec.env_var}=...\n"
                f"  (.env is gitignored. A plain export also works.)"
            )

    def _accepted_models(self) -> list[str]:
        """Models whose answer this call would accept, in preference order."""
        return [self.model] + [m for m in self._fallbacks if m != self.model]

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048,
                 variant: str = "") -> str:
        """One completion, served from cache when an identical request was made
        before.

        The cache is consulted for every model this call would accept, not just
        the requested one, because quota rotation means yesterday's answer to
        this exact prompt may sit under a fallback model's key. Checking only
        `self.model` would re-pay a 429 for a prompt already answered.

        `variant` distinguishes intentionally-repeated calls - see
        `cache.cache_key`; multi-query expansion needs k distinct samples of one
        prompt and would otherwise get the same cached answer k times.
        """
        if self.cache is not None:
            hit = self.cache.lookup_any(
                self._accepted_models(), self.spec.name, prompt, system,
                self.temperature, max_tokens, variant)
            if hit is not None:
                response, model, _key = hit
                self.last_model, self.last_cached = model, True
                return response

        t0 = time.perf_counter()
        response = self._complete_uncached(prompt, system, max_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.last_model, self.last_cached = self.model, False

        if self.cache is not None:
            # Stored under the model that actually answered, not the one that
            # was asked. `lookup_any` finds it either way, and a record whose
            # key and `model` field disagreed would be a provenance lie.
            self.cache.put(
                cache_key(self.spec.name, self.model, prompt, system,
                          self.temperature, max_tokens, variant),
                response, provider=self.spec.name, model=self.model,
                prompt=prompt, system=system, temperature=self.temperature,
                max_tokens=max_tokens, variant=variant, elapsed_ms=elapsed_ms)
        return response

    def _complete_uncached(self, prompt: str, system: str, max_tokens: int) -> str:
        fn = getattr(self, f"_{self.spec.name}")
        try:
            return fn(prompt, system, max_tokens)
        except QuotaExhausted:
            self._exhausted.add(self.model)
            for candidate in self._fallbacks:
                if candidate in self._exhausted:
                    continue
                self.model = candidate
                try:
                    return fn(prompt, system, max_tokens)
                except QuotaExhausted:
                    self._exhausted.add(candidate)
            raise QuotaExhausted(
                f"free-tier quota exhausted on every model tried: "
                f"{sorted(self._exhausted)}.\n"
                f"  Gemini resets daily. Options: wait, use a different key, "
                f"or --provider groq."
            ) from None

    def json_complete(self, prompt: str, system: str = "", max_tokens: int = 2048,
                      variant: str = "") -> object:
        """Complete and parse JSON, tolerating markdown fences.

        Models wrap JSON in ```json blocks regardless of instructions, often
        enough that stripping it here is cheaper than a retry loop.
        """
        raw = self.complete(prompt, system, max_tokens, variant).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model did not return JSON: {raw[:300]!r}") from exc

    # -- providers ---------------------------------------------------------

    def _gemini(self, prompt: str, system: str, max_tokens: int) -> str:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.key}")
        gen: dict = {"temperature": self.temperature, "maxOutputTokens": max_tokens}
        # Gemini 2.5+ are reasoning models: by default they spend output tokens
        # on hidden thinking before emitting anything, which truncated a JSON
        # array mid-generation at a 512-token budget. These tasks are short
        # structured generations with nothing to reason about, so the budget
        # goes to the answer instead.
        #
        # Ask for it whenever it might apply rather than matching on version
        # prefixes: "-latest" aliases silently move between versions, so a
        # name-based test would stop firing exactly when a new alias starts
        # pointing at a reasoning model. Older models reject the field with a
        # 400, which we detect and retry without it.
        gen["thinkingConfig"] = {"thinkingBudget": 0}

        payload: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        try:
            data = _post(url, payload, {"Content-Type": "application/json"},
                         model_hint=self.model)
        except LLMError as exc:
            if "HTTP 400" not in str(exc):
                raise
            gen.pop("thinkingConfig")
            data = _post(url, payload, {"Content-Type": "application/json"},
                         model_hint=self.model)
        try:
            cand = data["candidates"][0]
            parts = cand.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if "text" in p)
            if not text:
                # A safety block, or a reasoning model that spent the whole
                # budget on thinking tokens, returns a candidate with no text
                # part. Surface finishReason - it names which happened.
                raise LLMError(
                    f"empty response from {self.model} "
                    f"(finishReason={cand.get('finishReason')}). "
                    f"If MAX_TOKENS, raise max_tokens.")
            return text
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected gemini response: {json.dumps(data)[:400]}") from exc

    def _groq(self, prompt: str, system: str, max_tokens: int) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        data = _post(
            "https://api.groq.com/openai/v1/chat/completions",
            {"model": self.model, "messages": messages,
             "temperature": self.temperature, "max_tokens": max_tokens},
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        return data["choices"][0]["message"]["content"]

    def _anthropic(self, prompt: str, system: str, max_tokens: int) -> str:
        payload: dict = {"model": self.model, "max_tokens": max_tokens,
                         "temperature": self.temperature,
                         "messages": [{"role": "user", "content": prompt}]}
        if system:
            payload["system"] = system
        data = _post(
            "https://api.anthropic.com/v1/messages", payload,
            {"Content-Type": "application/json", "x-api-key": self.key,
             "anthropic-version": "2023-06-01"},
        )
        return "".join(b.get("text", "") for b in data.get("content", []))

    def _ollama(self, prompt: str, system: str, max_tokens: int) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "options": {"temperature": self.temperature, "num_predict": max_tokens}}
        if system:
            payload["system"] = system
        try:
            data = _post("http://localhost:11434/api/generate", payload,
                         {"Content-Type": "application/json"}, retries=1)
        except LLMError as exc:
            raise LLMError(
                f"ollama unreachable at localhost:11434 ({exc}).\n"
                f"  start it: ollama serve\n"
                f"  pull:     ollama pull {self.model}"
            ) from exc
        return data.get("response", "")
