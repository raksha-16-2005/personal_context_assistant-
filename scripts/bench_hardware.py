#!/usr/bin/env python
"""Measure embedding + rerank throughput on this machine.

Every performance claim in the README traces back to this script. It is the
evidence for why the ablations run on a subsample and why the shipped reranker
is not the one the plan originally specified - so it has to be re-runnable by
anyone reading the repo, on their own hardware, to see how the numbers move.

    make bench-hardware

Baseline recorded in docs/HARDWARE.md: MacBook Pro 2019, i7-9750H, CPU only.
"""
import json
import statistics
import time
from pathlib import Path

import torch

torch.set_num_threads(6)          # physical cores; hyperthreads hurt GEMM
torch.set_num_interop_threads(1)

BODY = (
    "Hi Mark, following up on the pricing discussion from Tuesday's call. Legal has "
    "reviewed the revised MSA and flagged two items in section 4.2 regarding the "
    "termination clause and the liability cap. I've attached the redline. Can you get "
    "me your comments before the steering committee meets next Thursday? Also, "
    "Jennifer asked whether we're still committed to the Q3 rollout date given the "
    "vendor delay - I told her we'd confirm by end of week. The other open question "
    "is whether the discount tier applies retroactively to the existing contracts or "
    "only to new business. Rick thought retroactive, but that contradicts what we "
    "agreed in the March memo. Let me know if you want to sync before then. Thanks, Sara"
)
DOCS = [f"Message {i}. {BODY}" for i in range(256)]
QUERY = "what did we decide about the pricing model and the discount tier"


def bench_embed(name, batch=32, n=256):
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    m = SentenceTransformer(name, device="cpu")
    load = time.time() - t0
    dim = m.get_sentence_embedding_dimension()
    m.encode(DOCS[:16], batch_size=batch, show_progress_bar=False)  # warmup
    runs = []
    for _ in range(3):
        t0 = time.time()
        m.encode(DOCS[:n], batch_size=batch, show_progress_bar=False)
        runs.append(n / (time.time() - t0))
    best, med = max(runs), statistics.median(runs)
    print(f"{name:32s} dim={dim:4d} load={load:5.1f}s  "
          f"{med:7.1f} docs/s (median)  {best:7.1f} peak")
    del m
    return med, dim


def bench_rerank(name, k=50):
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(name, device="cpu", max_length=512)
    pairs = [(QUERY, d) for d in DOCS[:k]]
    m.predict(pairs[:8], show_progress_bar=False)  # warmup
    runs = []
    for _ in range(3):
        t0 = time.time()
        m.predict(pairs, batch_size=16, show_progress_bar=False)
        runs.append(time.time() - t0)
    med = statistics.median(runs)
    print(f"{name:32s} top-{k} rerank = {med*1000:7.0f} ms/query  ({k/med:5.1f} pairs/s)")
    del m
    return med


print("=== EMBEDDING (email-length docs, ~230 tokens) ===")
res = {}
for mdl in ["sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5"]:
    try:
        res[mdl] = bench_embed(mdl)
    except Exception as e:
        print(f"{mdl}: FAILED {type(e).__name__}: {e}")

print("\n=== CROSS-ENCODER RERANK (latency per query) ===")
rerank_ms = {}
for mdl in ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"]:
    try:
        rerank_ms[mdl] = bench_rerank(mdl) * 1000
    except Exception as e:
        print(f"{mdl}: FAILED {type(e).__name__}: {e}")

print("\n=== PROJECTION: wall-clock to embed a corpus ===")
for n_chunks in (50_000, 375_000):
    print(f"\n  corpus = {n_chunks:,} chunks")
    for mdl, (rate, dim) in res.items():
        hrs = n_chunks / rate / 3600
        gb = n_chunks * dim * 4 / 1e9
        print(f"    {mdl.split('/')[-1]:22s} {hrs:6.2f} h/pass   "
              f"vectors={gb*1000:6.0f} MB   x4 chunking configs = {hrs*4:6.2f} h")

out = Path("runs/hardware.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({
    "torch": torch.__version__,
    "threads": torch.get_num_threads(),
    "embed_docs_per_sec": {m: round(r, 2) for m, (r, _) in res.items()},
    "embed_dim": {m: d for m, (_, d) in res.items()},
    "rerank_top50_ms": {m: round(v) for m, v in rerank_ms.items()},
}, indent=2))
print(f"\nwrote {out}")
