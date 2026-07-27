#!/usr/bin/env python
"""Export a cross-encoder to int8 ONNX, and prove the export was worth it.

The shipped retrieval path has a 200 ms budget. Measured torch latency for the
MiniLM cross-encoder is 962 ms at top-50 (docs/HARDWARE.md), so the ranking that
wins dimension 4 cannot be served as-is. Dynamic int8 quantization is the lever:
weights drop to 8-bit integers, activations are quantized on the fly, and AVX2
integer kernels do the rest.

    .venv/bin/python scripts/export_onnx_reranker.py            # MiniLM, top-20
    .venv/bin/python scripts/export_onnx_reranker.py --model BAAI/bge-reranker-base

Two things this script refuses to let slide:

*Quantization is a quality change, not just a speed change.* int8 rounding
perturbs the scores, which can reorder passages. So the export is verified by
scoring the same pairs with both backends and reporting Spearman rank
correlation and top-1 agreement. A fast reranker that ranks differently is a
different system and has to be re-benchmarked, not swapped in silently.

*avx2, not avx512_vnni.* This machine is a Coffee Lake i7-9750H: AVX2, no VNNI.
Quantizing for an instruction set the CPU does not have either fails at load or
silently falls back to a slower kernel, and the whole point here is the latency.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.index.rerank import MAX_LENGTH, onnx_dir  # noqa: E402

QUERY = "what did we decide about the pricing model and the discount tier"

# The passages have to be *distinct*, spanning clearly-relevant to clearly-
# irrelevant. An earlier version of this script scored 48 copies of one message
# and reported rho = 0.49, which looked like severe quantization damage and was
# really the two backends shuffling 48 tied scores in different orders. Rank
# correlation over near-ties measures float noise, not ranking behaviour.
PASSAGES = [
    # relevant
    "We agreed the discount tier applies retroactively to existing contracts, "
    "overriding the March memo. Pricing model stays volume-based.",
    "Confirming Rick's read: tier 2 pricing is retroactive to January for anyone "
    "already on the volume schedule. New business prices at the published rate.",
    "The pricing model discussion from Tuesday landed on three tiers with the "
    "discount applied at the account level rather than per order.",
    "Attached is the revised pricing grid. Note the discount tier thresholds "
    "moved from 500k to 400k annual spend after Jennifer's pushback.",
    "Per our discussion on the discount structure - we are not extending tier "
    "pricing to the resellers this year. Direct accounts only.",
    # adjacent: same thread, different subject
    "Legal has reviewed the revised MSA and flagged two items in section 4.2, "
    "the termination clause and the liability cap. Redline attached.",
    "Can you get me your comments on the MSA before the steering committee meets "
    "next Thursday? Jennifer wants the Q3 rollout date confirmed by end of week.",
    "The vendor delay pushes the Q3 rollout by roughly three weeks. I told "
    "Jennifer we would confirm the new date by Friday.",
    "Steering committee agenda: MSA redline, Q3 rollout date, headcount for the "
    "Houston desk. Thirty minutes, Thursday 2pm.",
    "Following up on Tuesday's call - I still owe you the summary of what legal "
    "said about indemnification. Working on it today.",
    # same domain, unrelated content
    "Gas nominations for the 15th are in. Volumes are flat against last week "
    "except for the Katy hub, which is up about 8 percent.",
    "The Calpine renewal is closing Friday. I picked it up from Sara when she "
    "moved desks - shout if you need the file before then.",
    "Counterparty credit review for the West desk is done. Two names moved to "
    "the watch list, nothing that affects existing positions.",
    "Please review the attached ISDA schedule before we send it back to their "
    "counsel. Sections 5 and 11 are the ones they changed.",
    "Confirming the transport capacity release for November. 12,000 dth/day "
    "at the posted index, term through March.",
    # clearly irrelevant
    "Cafeteria menu for the week of the 14th. Tuesday is taco day, Thursday is "
    "the barbecue that got rained out last month.",
    "Reminder: the building fire drill is at 10am tomorrow. Take the north "
    "stairwell and assemble in the visitor lot.",
    "IT is patching the mail servers Saturday night. Expect webmail to be "
    "unavailable from 11pm to about 3am.",
    "Open enrollment for benefits closes on the 30th. If you are not changing "
    "plans you do not need to do anything.",
    "Anyone want the last two tickets to Sunday's game? Section 112, face value, "
    "first person to reply gets them.",
    "Happy birthday to Diane - cake in the break room at 3pm, please come by.",
    "The parking garage is repaving levels 3 and 4 next week. Use the surface "
    "lot on Smith Street and keep your badge handy.",
    "New hire orientation moved to the 8th floor training room. Same time, "
    "9am, bring two forms of ID.",
    "Travel policy update: coach only on domestic flights under four hours, "
    "and hotel receipts are required for anything over $150 a night.",
]
PAIRS = PASSAGES


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without pulling in scipy.stats for one number."""
    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            mean_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = mean_rank
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra) ** 0.5
    var_b = sum((y - mean_b) ** 2 for y in rb) ** 0.5
    return cov / (var_a * var_b) if var_a and var_b else 0.0


# Mean chunk length in the built pilot index (config.json: mean_tokens 299.7 for
# thread_aware). Cross-encoder cost is quadratic-ish in sequence length, so
# timing 30-token passages understates the real thing by several times: an
# earlier version of this script measured 56 ms on short passages, against
# 962 ms in docs/HARDWARE.md for 512-token docs. Latency is measured at a
# realistic passage length or it is not a latency measurement.
TIMING_TOKENS = 300


def timing_passage(tokenizer, n_tokens: int = TIMING_TOKENS) -> str:
    """A passage of about `n_tokens` reference tokens, built from real text."""
    filler = " ".join(PASSAGES)
    ids = tokenizer.encode(filler, add_special_tokens=False)
    while len(ids) < n_tokens:
        filler = filler + " " + filler
        ids = tokenizer.encode(filler, add_special_tokens=False)
    return tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)


def time_scoring(reranker, n: int, repeats: int = 3) -> float:
    """Median latency for scoring n pairs at realistic passage length."""
    pairs = [timing_passage(reranker.tok)] * n
    reranker.score(QUERY, pairs[:4])                    # warm up the kernels
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        reranker.score(QUERY, pairs)
        runs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(runs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--top-k", type=int, default=20,
                    help="k to report latency at; the shipped path uses 20")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    out = onnx_dir(args.model)
    quantized = out / "model_quantized.onnx"
    if quantized.exists() and not args.force:
        print(f"cached: {quantized} (use --force to rebuild)")
    else:
        if out.exists():
            shutil.rmtree(out)
        print(f"exporting {args.model} -> ONNX ...")
        t0 = time.time()
        model = ORTModelForSequenceClassification.from_pretrained(args.model, export=True)
        model.save_pretrained(out)
        # The tokenizer travels with the export: a quantized model loaded with a
        # different tokenizer is a silent correctness bug, not a load error.
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(args.model).save_pretrained(out)
        print(f"  fp32 export in {time.time()-t0:.0f}s")

        print("quantizing (dynamic int8, avx2) ...")
        t0 = time.time()
        quantizer = ORTQuantizer.from_pretrained(out)
        # is_static=False: no calibration dataset, weights-only + dynamic
        # activations. per_channel=False is the conservative default; per-channel
        # is slightly more accurate and slower to load, and the accuracy check
        # below is what would justify changing it.
        quantizer.quantize(
            save_dir=out,
            quantization_config=AutoQuantizationConfig.avx2(is_static=False,
                                                            per_channel=False),
        )
        print(f"  quantized in {time.time()-t0:.0f}s")

    sizes = {p.name: p.stat().st_size / 1e6 for p in out.glob("*.onnx")}
    for name, mb in sorted(sizes.items()):
        print(f"  {name:28s} {mb:6.1f} MB")

    if args.skip_verify:
        return 0

    # -- does it still rank the same, and is it actually faster? ------------
    from emailrag.index.rerank import CrossEncoderReranker

    print(f"\nscoring {len(PAIRS)} pairs on both backends ...")
    torch_rr = CrossEncoderReranker.load(args.model)
    torch_scores = torch_rr.score(QUERY, PAIRS).tolist()
    torch_ms = time_scoring(torch_rr, args.top_k)
    del torch_rr

    onnx_rr = CrossEncoderReranker.load(args.model, onnx_int8=True)
    onnx_scores = onnx_rr.score(QUERY, PAIRS).tolist()
    onnx_ms = time_scoring(onnx_rr, args.top_k)

    rho = spearman(torch_scores, onnx_scores)
    top1_torch = max(range(len(PAIRS)), key=lambda i: torch_scores[i])
    top1_onnx = max(range(len(PAIRS)), key=lambda i: onnx_scores[i])
    top5_torch = sorted(range(len(PAIRS)), key=lambda i: -torch_scores[i])[:5]
    top5_onnx = sorted(range(len(PAIRS)), key=lambda i: -onnx_scores[i])[:5]
    overlap = len(set(top5_torch) & set(top5_onnx))

    print(f"\n{'':22s}{'torch fp32':>14s}{'onnx int8':>14s}")
    print(f"{'latency @ top-'+str(args.top_k):22s}{torch_ms:11.0f} ms{onnx_ms:11.0f} ms"
          f"   ({torch_ms/onnx_ms:.1f}x)")
    print(f"{'  at '+str(TIMING_TOKENS)+'-token passages':22s}"
          f"{'':14s}{'':14s}   CPU-specific, see docs/HARDWARE.md")
    print(f"{'top-1 passage':22s}{top1_torch:14d}{top1_onnx:14d}"
          f"   {'agree' if top1_torch == top1_onnx else 'DISAGREE'}")
    print(f"{'top-5 overlap':22s}{'':14s}{overlap:>13d}/5")
    print(f"{'spearman rho':22s}{'':14s}{rho:14.4f}")

    report = {
        "model": args.model,
        "top_k": args.top_k,
        "passage_tokens": TIMING_TOKENS,
        "torch_ms_median": round(torch_ms, 1),
        "onnx_int8_ms_median": round(onnx_ms, 1),
        "speedup": round(torch_ms / onnx_ms, 2) if onnx_ms else None,
        "spearman_rho": round(rho, 4),
        "top1_agree": top1_torch == top1_onnx,
        "top5_overlap": overlap,
        "size_mb": {k: round(v, 1) for k, v in sizes.items()},
        "note": "latency is CPU-specific - see docs/HARDWARE.md for the machine",
    }
    (out / "quantization_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out / 'quantization_report.json'}")

    if rho < 0.99:
        print("\nwarning: rho < 0.99 - int8 reordered enough that dimension 4 "
              "must be re-run with the ONNX arm, not inherited from the torch "
              "arm.", file=sys.stderr)
    if onnx_ms > 200:
        print(f"\nwarning: {onnx_ms:.0f} ms at top-{args.top_k} is still over the "
              f"200 ms budget - lower top_k or drop reranking from the shipped "
              f"path.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
