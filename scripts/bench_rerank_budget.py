#!/usr/bin/env python
"""What reranking configuration, if any, fits a 200 ms budget on this CPU.

`bench_hardware.py` established that the plan's reranker is unaffordable here:
962 ms at top-50 for MiniLM, 6.8 s for bge-reranker-base. The assumption was
that an int8 ONNX export would close the gap. It does not - see
`scripts/export_onnx_reranker.py`, which measures int8 as *slower* than torch
fp32 at realistic passage length on this machine (no AVX-512 VNNI, so the
dynamic quantize/dequantize overhead is not repaid by the integer kernels).

That leaves three levers, and this script measures all three together because
they trade against each other:

    k                how many chunks get reranked at all
    passage tokens   how much of each chunk the cross-encoder sees
    model depth      L-6 vs L-4 vs L-2 layers

Passage truncation is the interesting one. Attention is quadratic in sequence
length, so halving the passage roughly quarters the cost of that pair - and a
reranker deciding whether a 300-token email is about the discount tier rarely
needs the last 150 tokens. Unlike lowering k, truncation costs no recall: every
candidate still gets scored.

    make bench-rerank-budget

Latency here is hardware-specific and belongs only in tables labelled with this
CPU (docs/HARDWARE.md). Any quality claim about truncation has to come from
`make bench --dimension rerank`, not from here.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import transformers  # noqa: E402

transformers.logging.set_verbosity_error()      # the >512-token notice is expected

from emailrag.index.rerank import CrossEncoderReranker  # noqa: E402

BUDGET_MS = 200

QUERY = "what did we decide about the pricing model and the discount tier"

# Realistic chunk length: the built pilot index reports mean_tokens = 299.7.
PASSAGE_TOKENS = (128, 192, 300, 512)
TOP_KS = (10, 20, 50)

MODELS = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-4-v2",
    "cross-encoder/ms-marco-MiniLM-L-2-v2",
)

FILLER = (
    "Confirming Rick's read: tier 2 pricing is retroactive to January for anyone "
    "already on the volume schedule, and new business prices at the published "
    "rate. Legal reviewed the revised MSA and flagged section 4.2, the "
    "termination clause and the liability cap. The steering committee meets "
    "Thursday and Jennifer wants the Q3 rollout date confirmed by end of week. "
    "Gas nominations for the fifteenth are flat except the Katy hub. "
)


def passage(tokenizer, n_tokens: int) -> str:
    text = FILLER
    while len(tokenizer.encode(text, add_special_tokens=False)) < n_tokens:
        text += FILLER
    ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def measure(reranker, n_pairs: int, text: str, repeats: int = 3) -> float:
    pairs = [text] * n_pairs
    reranker.score(QUERY, pairs[:4])                     # warm the kernels
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        reranker.score(QUERY, pairs)
        runs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(runs)


def main() -> int:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)

    rows: list[dict] = []
    for model_id in MODELS:
        try:
            rr = CrossEncoderReranker.load(model_id, batch_size=16)
        except Exception as exc:                          # noqa: BLE001
            print(f"{model_id}: unavailable ({type(exc).__name__}: {exc})")
            continue

        short = model_id.split("/")[-1].replace("ms-marco-MiniLM-", "")
        print(f"\n=== {model_id} ===")
        print(f"{'passage tokens':>15s} " + "".join(f"{'top-'+str(k):>12s}" for k in TOP_KS))
        for n_tokens in PASSAGE_TOKENS:
            text = passage(rr.tok, n_tokens)
            cells = []
            for k in TOP_KS:
                ms = measure(rr, k, text)
                rows.append({"model": model_id, "passage_tokens": n_tokens,
                             "top_k": k, "ms": round(ms, 1),
                             "within_budget": ms <= BUDGET_MS})
                mark = "*" if ms <= BUDGET_MS else " "
                cells.append(f"{ms:10.0f}{mark} ")
            print(f"{n_tokens:>15d} " + "".join(cells))
        del rr

    print(f"\n* = within the {BUDGET_MS} ms shipped-path budget")

    affordable = [r for r in rows if r["within_budget"]]
    if affordable:
        # Prefer the arm that scores the most chunks at the most tokens: k first,
        # then passage length. Latency is the constraint, quality is the goal,
        # and quality is monotone in both.
        best = max(affordable, key=lambda r: (r["top_k"], r["passage_tokens"]))
        print(f"\nbest affordable arm: {best['model'].split('/')[-1]} "
              f"top-{best['top_k']} @ {best['passage_tokens']} tokens = "
              f"{best['ms']:.0f} ms")
        print("Quality for this arm is not established here - run "
              "`make bench --dimension rerank` before shipping it.")
    else:
        print(f"\nNothing fits {BUDGET_MS} ms. The shipped path reranks nothing, "
              f"or the budget moves.")

    out = Path("runs/rerank_budget.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "budget_ms": BUDGET_MS,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "note": "CPU-specific; see docs/HARDWARE.md for the machine",
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
