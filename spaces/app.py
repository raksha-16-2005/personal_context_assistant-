"""Hugging Face Space demo - ZeroGPU Gradio app.

Deployed separately from the local UI, and using a different framework on purpose.
`scripts/serve.py` is stdlib-only because this repo's `requirements.txt` is pinned
to torch 2.2.2 for Intel macOS, and Gradio needs newer `huggingface_hub` and
`pydantic` than that tolerates. A Space is a fresh Linux environment where the pin
does not apply, so Gradio is the right tool here and the wrong one there.

**ZeroGPU, not CPU.** Hugging Face now requires a paid plan for ordinary
Gradio/Docker Spaces; free personal accounts get two ZeroGPU Spaces. That is the
only free route, and it changes the code: `@spaces.GPU` marks the functions that
may touch the accelerator, and the GPU is attached per call rather than held.

**This Space is Enron-only, permanently.** The retrieval index is public-corpus
data. Phase 7 points the same pipeline at a private mailbox, and that index must
never be uploaded here - see `check_public_corpus()`, which refuses to start if it
finds a non-Enron index. A demo over private mail is also one nobody else can
reproduce, which is the whole reason the Enron half exists.

**Latency shown here is not the project's latency.** Every timing table in the
README is measured on one documented CPU (docs/HARDWARE.md). A ZeroGPU runner is
different hardware with a cold-start penalty, so the UI labels its numbers as
Space-specific rather than letting them be read as the system's.

Files a Space needs beside this one: `requirements.txt` (in this directory, NOT the
repo root) and `README.md` with the Space YAML header.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import gradio as gr

# Repo layout: this file lives in spaces/, the package in src/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import spaces                      # provided by the ZeroGPU runtime
    HAS_ZEROGPU = True
except ImportError:                    # local `python spaces/app.py` for a smoke test
    HAS_ZEROGPU = False

    class _Stub:
        """No-op `spaces.GPU`, so the app can be smoke-tested off-platform.

        Handles both decorator forms: bare `@spaces.GPU` passes the function as the
        first positional argument, `@spaces.GPU(duration=60)` passes none.
        """

        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda fn: fn

    spaces = _Stub()                   # type: ignore[assignment]

INDEX_ROOT = Path(os.environ.get("EMAILRAG_INDEX_ROOT", ROOT / "data" / "index"))
SAMPLE = Path(os.environ.get("EMAILRAG_SAMPLE", ROOT / "data" / "interim" / "sample.parquet"))

# Enron owners that must appear in the corpus for it to be the public dataset. A
# private index would not contain them.
ENRON_MARKERS = ("@enron.com",)

EXAMPLES = [
    "what was decided about the confidentiality agreement",
    "what are the terms of the Dynegy merger",
    "who is handling the California power crisis filings",
    "who won the 2030 world cup",
]

PIPELINE = None


def check_public_corpus(sample: Path) -> None:
    """Refuse to serve anything but the public Enron corpus.

    Phase 7 aims this pipeline at a real mailbox, and the failure mode - a private
    index pushed to a public Space - is unrecoverable once it happens. So the check
    is a startup assertion rather than a note in a README.
    """
    import pyarrow.parquet as pq

    senders = pq.read_table(sample, columns=["sender"]).column("sender").to_pylist()
    sample_of = [s for s in senders[:5000] if s]
    if not any(any(m in (s or "") for m in ENRON_MARKERS) for s in sample_of):
        raise SystemExit(
            "refusing to start: no Enron senders found in the corpus, so this may "
            "be private mail. The public Space is Enron-only - see phase 7 in "
            "TODO.md. Set EMAILRAG_SAMPLE to the public corpus.")


def load_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    from emailrag.pipeline import DEFAULT_RERANK, Pipeline

    built = sorted(d for d in INDEX_ROOT.iterdir()
                   if d.is_dir() and (d / "config.json").exists()) if INDEX_ROOT.exists() else []
    if not built:
        raise SystemExit(
            f"no built index under {INDEX_ROOT}. A Space needs the index committed "
            f"or pulled from a Dataset at startup - see spaces/README.md.")
    check_public_corpus(SAMPLE)

    PIPELINE = Pipeline(built[0], SAMPLE, rerank=DEFAULT_RERANK, verbose=True)
    return PIPELINE


def format_sources(messages, cited: list[int]) -> str:
    used = set(cited or [])
    blocks = []
    for m in messages:
        mark = "**cited**" if m["n"] in used else "not cited"
        blocks.append(
            f"### [{m['n']}] {m['subject'] or '(no subject)'}  ·  {mark}\n"
            f"`{m['sender']}` → `{(m['recipients'] or '')[:70]}`  ·  "
            f"{m['date'] or 'undated'}  ·  score {m['score']}\n\n"
            f"> {' '.join((m['text'] or '').split())[:700]}\n")
    return "\n".join(blocks) or "_nothing retrieved_"


# The GPU is attached per call, not held for the life of the Space. Reranking and
# query encoding are the only parts that would use it.
@spaces.GPU(duration=60)
def answer(question: str, n_sources: int, rerank_arm: str, search_only: bool):
    question = (question or "").strip()
    if not question:
        return "Ask something.", "", ""

    pipe = load_pipeline()
    from emailrag.index import rerank as RR

    if rerank_arm in RR.SPECS and rerank_arm != pipe.rerank:
        spec = RR.SPECS[rerank_arm]
        pipe.rerank = rerank_arm
        pipe.reranker = RR.CrossEncoderReranker.from_spec(spec) if spec else None
        pipe.rerank_top_k = spec.top_k if spec else 0

    t0 = time.perf_counter()
    if search_only:
        result = pipe.search(question, top_n=10)
        answer_md = ("_Search only - no answer generated._\n\n"
                     "Retrieval ran; nothing was sent to a language model.")
        cited: list[int] = []
    else:
        ans, result = pipe.ask(question, n_sources=int(n_sources), top_n=10)
        cited = ans.cited_numbers
        if ans.error:
            answer_md = (f"**Generation failed:** {ans.error}\n\n"
                         f"Retrieval still worked - the sources are below.")
        elif ans.refused:
            answer_md = (
                "**No answer in this corpus.** The model returned "
                "`INSUFFICIENT_CONTEXT` rather than guessing, which is the correct "
                "outcome for a question these emails cannot answer. Refusal is a "
                "measured metric here, not a failure.")
        else:
            answer_md = ans.text
            if ans.invalid_citations:
                answer_md += (f"\n\n> ⚠️ **Fabricated citation(s)** "
                              f"{ans.invalid_citations} — only "
                              f"{len(ans.citations)} sources were supplied.")
            if ans.uncited_sentences:
                answer_md += ("\n\n> ⚠️ **Uncited claim(s):** "
                              + " / ".join(s[:90] for s in ans.uncited_sentences))
    wall = (time.perf_counter() - t0) * 1000

    messages = [{"n": i + 1, "message_id": m.message_id, "sender": m.sender,
                 "recipients": m.recipients, "date": m.date, "subject": m.subject,
                 "score": round(m.score, 4), "text": m.text}
                for i, m in enumerate(result.messages)]

    t = result.timings
    timing_md = (
        f"retrieval {t.get('retrieval_ms', 0):.0f} ms · "
        f"rerank {t.get('rerank_ms', 0):.0f} ms · total {wall:.0f} ms\n\n"
        f"_Measured on this "
        f"{'ZeroGPU' if HAS_ZEROGPU else 'local'} runner, which is not the machine "
        f"the README's latency tables come from. Those are one documented CPU - see "
        f"docs/HARDWARE.md - and mixing platforms in one table is exactly what this "
        f"project refuses to do._")

    return answer_md, format_sources(messages, cited), timing_md


def build_ui() -> gr.Blocks:
    from emailrag.index import rerank as RR

    with gr.Blocks(title="Email retrieval", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Email retrieval over the Enron corpus\n"
            "Hybrid retrieval (dense + BM25, reciprocal-rank fusion) with a "
            "cross-encoder rerank, then an answer built **only** from the retrieved "
            "excerpts, with every claim cited.\n\n"
            "Ask something the corpus cannot answer and it returns "
            "`INSUFFICIENT_CONTEXT` instead of guessing — refusal rate on ten "
            "unanswerable controls is one of the reported metrics.\n\n"
            "*Public Enron data only. The same pipeline runs over a private mailbox "
            "locally; that index is never uploaded here.*")

        with gr.Row():
            question = gr.Textbox(label="Question", scale=5,
                                  placeholder="what was decided about the confidentiality agreement")
            ask_btn = gr.Button("Ask", variant="primary", scale=1)

        with gr.Accordion("Configuration", open=False):
            with gr.Row():
                n_sources = gr.Slider(3, 10, value=6, step=1,
                                      label="sources given to the generator")
                rerank_arm = gr.Dropdown(sorted(RR.SPECS), value="L2@20/t192",
                                         label="rerank arm")
                search_only = gr.Checkbox(False, label="search only (no LLM call)")

        answer_md = gr.Markdown(label="Answer")
        timing_md = gr.Markdown()
        with gr.Accordion("Retrieved sources", open=True):
            sources_md = gr.Markdown()

        gr.Examples(EXAMPLES, inputs=question, label="Try one")

        inputs = [question, n_sources, rerank_arm, search_only]
        outputs = [answer_md, sources_md, timing_md]
        ask_btn.click(answer, inputs=inputs, outputs=outputs)
        question.submit(answer, inputs=inputs, outputs=outputs)

    return demo


if __name__ == "__main__":
    build_ui().queue().launch()
