"""Content-addressed disk cache for LLM responses.

This exists because of an arithmetic problem, not for elegance. Dimension 5
(query transformation), the router, the extraction arms and the generation
judge all call an LLM *per query per config*. At 80 queries x 6 index configs x
4 retrievers, one `make bench` is thousands of calls - several times the Gemini
free-tier daily quota. Without a cache the second run of the day cannot happen,
which means the tables cannot be regenerated, which breaks the one claim this
project actually makes: that `make bench` reproduces every number from the
published eval set.

Three design points worth defending:

*Keyed on the exact request, including the model.* A cached Gemini answer must
never be served for an Ollama call, and the extraction comparison is
specifically local-model-vs-hosted-model - silently crossing those wires would
fabricate the headline result. The key covers provider, model, temperature,
max_tokens, system prompt and user prompt. Change any of them and it is a
different request.

*Rotation-aware lookups.* `LLM` rotates through Gemini models when one model's
daily quota is spent, so the model that answered a given prompt yesterday may
not be the model requested today. `lookup_any` checks every model the caller
would accept, in preference order, before any network call - otherwise the
first re-run re-pays a 429 for every prompt it already has an answer for.

*Nothing is cached unless it succeeded.* A cached error is poison: it would
persist past the transient condition that caused it and be indistinguishable
from a real response. Failures are simply not written.

The cache is deliberately unbounded and has no TTL. It is an experiment log:
staleness is the point, because a benchmark that quietly re-queries a
newer model version stops being reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Bumped only when the on-disk record shape changes. The prompt is already part
# of the key, so prompt edits do not need a version bump - they are new keys.
CACHE_VERSION = 1

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "llm_cache"

# Kill switch for a deliberately fresh call - measuring real latency, or
# checking whether a prompt fix actually changed anything.
ENV_DISABLE = "EMAILRAG_LLM_CACHE"


def disabled_by_env() -> bool:
    """Read the kill switch on every access rather than at construction.

    The process-wide cache in `client.default_cache()` is built lazily and then
    memoised, so a construction-time check would freeze whatever the
    environment happened to say at the moment of the first LLM call - which in
    a test run is whichever test file imported first.
    """
    return os.environ.get(ENV_DISABLE, "").strip().lower() in {"0", "false", "no"}


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(provider: str, model: str, prompt: str, system: str,
              temperature: float, max_tokens: int, variant: str = "") -> str:
    """Stable hash of everything that can change a response.

    `variant` is the escape hatch for intentionally-diverse sampling. Multi-query
    expansion asks for k rewrites of one query at temperature > 0; with the
    prompt alone as the key, all k samples would collapse onto the first cached
    answer and the expansion would silently degrade to a single query. Callers
    that want n distinct samples pass the sample index here, which keeps them
    distinct *and* reproducible.
    """
    return hashlib.sha256(_canonical({
        "v": CACHE_VERSION,
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "system": system,
        "temperature": round(float(temperature), 4),
        "max_tokens": int(max_tokens),
        "variant": variant,
    }).encode()).hexdigest()


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def calls(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.calls if self.calls else 0.0

    def render(self) -> str:
        return (f"llm cache: {self.hits}/{self.calls} hits "
                f"({self.hit_rate:.0%}), {self.writes} written")


@dataclass
class ResponseCache:
    """One directory of JSON records, one record per distinct request."""

    root: Path = field(default_factory=lambda: DEFAULT_DIR)
    enabled: bool = True
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def active(self) -> bool:
        """Enabled here *and* not killed by the environment, so
        `EMAILRAG_LLM_CACHE=0 make bench` needs no code path of its own."""
        return self.enabled and not disabled_by_env()

    # -- paths -------------------------------------------------------------

    def _path(self, key: str) -> Path:
        # Two-character fan-out: a flat directory of tens of thousands of files
        # is slow to list and unpleasant to inspect by hand.
        return self.root / key[:2] / f"{key}.json"

    # -- reads -------------------------------------------------------------

    def get(self, key: str) -> str | None:
        """Return the cached completion, or None. Does not touch `stats` -
        `lookup_any` owns hit/miss accounting so one logical call counts once."""
        if not self.active:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # A record truncated by a kill during write is a miss, not a crash.
            return None
        response = record.get("response")
        return response if isinstance(response, str) else None

    def lookup_any(self, models: list[str], provider: str, prompt: str, system: str,
                   temperature: float, max_tokens: int,
                   variant: str = "") -> tuple[str, str, str] | None:
        """First hit across `models`, in order. Returns (response, model, key).

        The key returned is always the *first* model's key, i.e. what a fresh
        call would write, so a caller can store under the request it made while
        still being served an answer an accepted fallback produced earlier.
        """
        keys = [cache_key(provider, m, prompt, system, temperature, max_tokens, variant)
                for m in models]
        for model, key in zip(models, keys):
            hit = self.get(key)
            if hit is not None:
                self.stats.hits += 1
                return hit, model, key
        self.stats.misses += 1
        return None

    # -- writes ------------------------------------------------------------

    def put(self, key: str, response: str, *, provider: str, model: str,
            prompt: str, system: str = "", temperature: float = 0.0,
            max_tokens: int = 0, variant: str = "",
            elapsed_ms: float | None = None) -> None:
        """Write one record atomically.

        Provenance is stored alongside the response - which model produced it
        and when - because a cache that cannot answer "where did this number
        come from" is not an experiment log. Prompts are stored in full: they
        are what makes a surprising cached answer debuggable.
        """
        if not self.active:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "v": CACHE_VERSION,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "variant": variant,
            "system": system,
            "prompt": prompt,
            "response": response,
            "cached_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_ms": round(elapsed_ms, 1) if elapsed_ms is not None else None,
        }
        # tempfile + replace: two `make bench` processes on the same cache must
        # never leave a half-written record behind for the other to read.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(record, fh, ensure_ascii=False)
            os.replace(tmp, path)
            self.stats.writes += 1
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- housekeeping ------------------------------------------------------

    def __len__(self) -> int:
        if not self.root.exists():
            return 0
        return sum(1 for _ in self.root.glob("*/*.json"))

    def size_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(p.stat().st_size for p in self.root.glob("*/*.json"))

    def clear(self) -> int:
        n = 0
        for path in self.root.glob("*/*.json"):
            path.unlink()
            n += 1
        return n
