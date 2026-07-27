# Hardware profile and what it forces

Every performance number in this repo was measured on one machine. Stating it
plainly is part of the claim: a latency figure without the hardware behind it
is not reproducible.

```
MacBook Pro (2019)
Intel Core i7-9750H @ 2.60 GHz - 6 physical cores / 12 threads, AVX2
16 GB RAM
Intel UHD 630 + Radeon Pro 555X   <- unusable for training or inference
macOS 15.7.4, x86_64
```

**There is no GPU path.** CUDA is absent and PyTorch's MPS backend requires
Apple silicon, so the AMD dGPU cannot be used. Everything below is CPU.

## The toolchain constraint

PyTorch stopped publishing `macosx_x86_64` wheels after **2.2.2**; the current
release is 2.13.0. Anything newer simply has no wheel for this platform.

`transformers >= 4.46` hard-requires `torch >= 2.4`. Install it against torch
2.2.2 and it does not error - it *silently disables the torch backend*:

```
[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.2
[transformers] PyTorch was not found. Models won't be available ...
```

The pins in `requirements.txt` are the newest combination verified working
here. Do not bump torch or transformers without re-verifying on Intel macOS.

**Interpreter:** use Homebrew's `python3.11` (`/usr/local/bin/python3.11`),
not pyenv's. The pyenv 3.11.9 build on this machine was compiled without
`liblzma`, so `import lzma` fails; `datasets` imports it, `sentence-transformers`
imports `datasets`, and the whole stack fails to load. `make venv` pins the
correct interpreter.

## Measured throughput

Embedding, on email-length documents (~230 tokens), 6 threads:

| Model | dim | docs/s | 50k chunks | 375k chunks |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 67.2 | 0.21 h | 1.55 h |
| bge-small-en-v1.5 | 384 | 34.5 | 0.40 h | 3.02 h |
| bge-base-en-v1.5 | 768 | 9.8 | 1.41 h | 10.58 h |

### Correction: the microbenchmark is 2.6x optimistic

A full index build measured **25.8 docs/s** for MiniLM, against 67.2 in the
microbenchmark above — 57,997 chunks in 37 minutes.

Chunk length explains only part of it. Mean chunk is **300 tokens** versus 230
in the benchmark, which is a 1.3x difference against a 2.6x slowdown. The rest
is sustained-load behaviour the microbenchmark cannot see: a 9750H in a 2019
chassis throttles hard over a 37-minute all-core run, where the benchmark's
three-second bursts run at full boost. (A concurrent 2-thread job during part
of this build took some share too.)

**Plan multi-hour work against the sustained column.** A short benchmark on
this machine will always flatter you.

| Model | burst docs/s | sustained docs/s | one pass over 58k chunks |
|---|---|---|---|
| all-MiniLM-L6-v2 | 67.2 | **25.8** (measured) | **37 min** (measured) |
| bge-small-en-v1.5 | 34.5 | ~13 (projected) | ~1.2 h |
| bge-base-en-v1.5 | 9.8 | ~3.8 (projected) | ~4.2 h |

The 50k-message sample yields **57,997 chunks** under thread-aware chunking —
1.16 per message. Thread-aware splits long messages more often than it packs
short ones on this corpus, so it does not reduce chunk count the way the name
suggests.

Cross-encoder rerank latency, per query:

| Model | top-50 |
|---|---|
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 962 ms |
| BAAI/bge-reranker-base | 6802 ms |

Reproduce with `make bench-hardware`.

## What those numbers changed in the plan

**Ablations run on a 50k-message stratified subsample, not the full corpus.**
At full scale bge-base alone across four chunking configs is tens of hours -
before thermal throttling, which is real on a 9750H over multi-hour all-core
runs, and before any re-run forced by a bug. An IR ablation needs the gold
documents plus enough distractors, not the whole corpus. The final chosen
configuration is built once over the full corpus so the headline number is
honest.

**Dimensions are varied one at a time, not as a grid.** With the corrected
throughput above, a 4-chunking x 3-model cross product is ~20.5 h of embedding.
Varying each dimension independently around a baseline - 4 chunkings against a
cheap model, then 2 more models against the winning chunking - is 6 builds and
**~6.6 h**, one overnight run. This is also what the plan actually asks for
("six dimensions, each varied independently"); a grid would additionally report
interaction effects that 70 answerable queries cannot resolve.

**Shipped rerank is top-20 with MiniLM + ONNX int8, not top-50 with
bge-reranker-base.** 6.8 s/query is not a shippable latency. bge-reranker-base
stays in the ablation as an offline quality ceiling.

**Extraction is scoped to the threads the temporal and entity-scoped eval
queries touch (~2-3k messages), not the whole corpus.** Ollama has no GPU
offload on Intel Macs; Qwen 2.5 3B Q4 runs ~8-12 tok/s, about 25 s per
message. That budget is one overnight run, and it is all the router needs.

**Postgres runs locally, not on Neon.** The Neon free plan caps storage at
0.5 GB per project; 375k 768-dim float32 vectors are 1.15 GB before any text
or index. Local Homebrew `postgresql@15` + pgvector is genuinely $0.

**The demo Space must use the ZeroGPU path.** Hugging Face now requires a paid
plan for Gradio/Docker Spaces; only Static Spaces are free. Free personal
accounts may host up to 2 Gradio Spaces on ZeroGPU, which is the route here.

## Operating notes

- `maintenance_work_mem` is raised **per session** during index builds, never
  globally - this Postgres server hosts unrelated project databases.
- pgvector is built from source against `postgresql@15` with `-march=native`;
  the bottle only ships for pg 17/18. The `.so` is therefore tied to this CPU.
- Load models **sequentially**. Holding bge-base, a cross-encoder, Ollama and
  Postgres at once will swap on 16 GB.
- Budget ~15 GB disk: models ~2 GB (bge-reranker-base alone is 1.1 GB), raw
  maildir ~2.6 GB, plus indices.
