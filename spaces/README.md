---
title: Email Retrieval RAG
emoji: 📧
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
short_description: Hybrid RAG over the Enron corpus, with cited answers and measured refusal
---

# Email retrieval over the Enron corpus

Hybrid retrieval (dense + BM25, reciprocal-rank fusion) with a cross-encoder
rerank, then an answer synthesised **only** from the retrieved excerpts, with every
claim carrying a citation back to the message it came from.

Ask it something the corpus cannot answer and it returns `INSUFFICIENT_CONTEXT`
instead of guessing. Refusal rate over ten deliberately unanswerable controls is a
reported metric, not a hope.

Source, evaluation methodology, and the ablation tables:
**[github.com/<user>/personal-context-assistant](https://github.com/)**

## Deploying this Space

This directory is *not* the Space repo — it is the source of one. A Space needs a
flat repo with `app.py` at the root, so the deploy copies this directory's contents
plus the `src/emailrag` package and the built index.

```bash
# 1. Create the Space (ZeroGPU - free personal accounts get two).
#    Plain Gradio/Docker Spaces are no longer free; ZeroGPU is the only free route.
huggingface-cli repo create email-retrieval-rag --type space \
    --space_sdk gradio --space_hardware zero-a10g

# 2. Assemble a flat Space repo.
git clone https://huggingface.co/spaces/<user>/email-retrieval-rag /tmp/space
cp spaces/app.py spaces/requirements.txt spaces/README.md /tmp/space/
cp -r src/emailrag /tmp/space/src/emailrag        # app.py expects src/ on the path

# 3. The index. It is ~230 MB for one config, which is over git's comfort but fine
#    for LFS or, better, a Dataset the Space pulls at startup.
cd /tmp/space && git lfs track "*.npy" "*.parquet" "*.jsonl.gz"
mkdir -p data/index data/interim
cp -r <repo>/data/index/thread_aware__all-MiniLM-L6-v2 data/index/
cp <repo>/data/interim/sample.parquet data/interim/

git add -A && git commit -m "Deploy" && git push
```

Set `EMAILRAG_INDEX_ROOT` / `EMAILRAG_SAMPLE` in the Space settings if the data
lands somewhere other than `data/`.

## Two things this Space deliberately does not do

**It never serves private mail.** `check_public_corpus()` refuses to start unless
Enron senders are present in the corpus. The same pipeline runs over a real mailbox
locally (phase 7 of the project's TODO), and that index is never uploaded — a demo
over private mail is also one nobody else could reproduce or verify, which is the
whole reason the Enron half exists.

**It does not report the project's latency numbers.** Every timing table in the
main README is measured on one documented CPU. A ZeroGPU runner is different
hardware with a cold-start penalty, so this UI labels its own timings as
Space-specific. Quality metrics (recall, MRR, nDCG) are hardware-independent and
portable; latency is not, and mixing platforms inside one table is precisely what
the project refuses to do.

## Why Gradio here and not in the repo

The local UI (`make serve`) is stdlib `http.server` with no new dependencies,
because the repo is pinned to torch 2.2.2 — the last version with an Intel-macOS
wheel — and Gradio requires newer `huggingface_hub` and `pydantic` than that stack
tolerates. A Space is a fresh Linux environment where the pin does not apply, so
Gradio is the right tool here and the wrong one there.
