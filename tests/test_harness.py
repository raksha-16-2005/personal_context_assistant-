from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.evaluation import harness as H
from emailrag.evaluation.evalset import EvalQuery
from emailrag.index.dense import DenseIndex
from emailrag.index.sparse import SparseIndex


class FakeModel:
    """Encodes text to a 3-d vector by keyword presence.

    Deterministic and instant. Loading a real sentence-transformer here would
    make the suite take minutes and would test HuggingFace rather than the
    harness wiring.
    """
    KEYWORDS = ("pricing", "audit", "lunch")

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kw):
        rows = []
        for t in texts:
            v = np.array([1.0 if k in t.lower() else 0.0 for k in self.KEYWORDS],
                         dtype=np.float32)
            if not v.any():
                v = np.array([0.33, 0.33, 0.33], dtype=np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows).astype(np.float32)


DOCS = {
    "m1:0": "pricing model discount tier decision",
    "m1:1": "pricing follow up from the same message",
    "m2:0": "quarterly audit findings houston",
    "m3:0": "lunch plans friday",
}


@pytest.fixture
def indices():
    ids = list(DOCS)
    texts = [DOCS[i] for i in ids]
    model = FakeModel()
    return (SparseIndex.build(ids, texts),
            DenseIndex(ids, model.encode(texts)),
            model)


def _q(qid, text, cls, rel):
    return EvalQuery(qid, text, cls, list(rel), verified=True)


def test_every_retriever_produces_a_scored_row(indices):
    bm25, dense, model = indices
    queries = [_q("q1", "pricing decision", "semantic", ["m1"])]

    for retriever in H.RETRIEVERS:
        row = H.run(H.RunConfig("thread_aware", "fake", retriever),
                    queries, bm25, dense, model)
        assert row["config"]["retriever"] == retriever
        assert row["overall"]["n_answerable"] == 1
        assert 0.0 <= row["overall"]["ndcg@10"] <= 1.0


def test_chunks_collapse_to_messages_before_scoring(indices):
    # m1 has two chunks. A perfect ranking must score 1.0, not be diluted by
    # the second chunk occupying a slot.
    bm25, dense, model = indices
    row = H.run(H.RunConfig("thread_aware", "fake", "bm25"),
                [_q("q1", "pricing model discount", "semantic", ["m1"])],
                bm25, dense, model)

    assert row["overall"]["recall@5"] == 1.0
    assert row["overall"]["mrr"] == 1.0
    assert row["per_query"][0]["found_at"] == 1


def test_unanswerable_controls_excluded_from_means(indices):
    bm25, dense, model = indices
    queries = [_q("q1", "pricing decision", "semantic", ["m1"]),
               _q("q2", "who won the 2030 world cup", "unanswerable", [])]
    row = H.run(H.RunConfig("thread_aware", "fake", "bm25"), queries, bm25, dense, model)

    assert row["overall"]["n_queries"] == 2
    assert row["overall"]["n_answerable"] == 1
    assert "unanswerable" in row["by_class"]


def test_per_class_breakdown_is_emitted(indices):
    bm25, dense, model = indices
    queries = [_q("q1", "pricing decision", "semantic", ["m1"]),
               _q("q2", "audit findings", "entity", ["m2"])]
    row = H.run(H.RunConfig("thread_aware", "fake", "bm25"), queries, bm25, dense, model)

    assert set(row["by_class"]) == {"semantic", "entity"}
    assert row["by_class"]["entity"]["n_answerable"] == 1


def test_latency_is_recorded_per_query(indices):
    bm25, dense, model = indices
    row = H.run(H.RunConfig("thread_aware", "fake", "dense"),
                [_q("q1", "pricing", "semantic", ["m1"])], bm25, dense, model)

    assert row["overall"]["p95_ms"] >= 0
    assert row["per_query"][0]["latency_ms"] >= 0


def test_found_at_is_none_when_nothing_relevant_is_retrieved(indices):
    bm25, dense, model = indices
    row = H.run(H.RunConfig("thread_aware", "fake", "bm25"),
                [_q("q1", "pricing", "semantic", ["nonexistent"])], bm25, dense, model)

    assert row["per_query"][0]["found_at"] is None
    assert row["overall"]["recall@5"] == 0.0


def test_unknown_retriever_is_rejected(indices):
    bm25, dense, model = indices
    with pytest.raises(ValueError, match="unknown retriever"):
        H.run(H.RunConfig("thread_aware", "fake", "telepathy"),
              [_q("q1", "pricing", "semantic", ["m1"])], bm25, dense, model)


def test_weight_sweep_covers_each_weight(indices):
    bm25, dense, model = indices
    rows = H.sweep_weights(H.RunConfig("thread_aware", "fake", "hybrid_weighted"),
                           [_q("q1", "pricing", "semantic", ["m1"])],
                           bm25, dense, model, weights=(0.3, 0.7))

    assert [r["config"]["dense_weight"] for r in rows] == [0.3, 0.7]


def test_retrieve_depth_exceeds_the_metric_cutoff():
    # recall@20 is meaningless if fewer than 20 distinct messages survive the
    # chunk->message collapse.
    assert H.RETRIEVE_DEPTH >= 100


def test_markdown_table_renders_one_row_per_config(indices):
    bm25, dense, model = indices
    queries = [_q("q1", "pricing", "semantic", ["m1"])]
    rows = [H.run(H.RunConfig("thread_aware", "fake", r), queries, bm25, dense, model)
            for r in ("bm25", "dense")]

    table = H.markdown_table(rows, group_by="retriever")
    lines = table.splitlines()

    assert lines[0].startswith("| Retriever")
    assert len(lines) == 4                 # header + separator + 2 rows
    assert "bm25" in table and "dense" in table


# -- dimension 4: reranking -------------------------------------------------

class OracleReranker:
    """Puts the chunk whose text contains `target` first. No model involved."""

    def __init__(self, target: str):
        self.target = target
        self.max_length = 512

    def rerank(self, query, ranked, texts, top_k):
        head, tail = ranked[:top_k], ranked[top_k:]
        head = sorted(head, key=lambda kv: self.target not in texts.get(kv[0], ""))
        return [(cid, float(-i)) for i, (cid, _) in enumerate(head + tail)]

    def rerank_timed(self, query, ranked, texts, top_k):
        return self.rerank(query, ranked, texts, top_k), 1.5


def test_reranking_changes_the_ranking_and_is_recorded(indices):
    bm25, dense, model = indices
    # "lunch" ranks first for this query on bm25; the reranker knows better.
    row = H.run(H.RunConfig("thread_aware", "fake", "bm25", rerank="oracle"),
                [_q("q1", "lunch", "semantic", ["m1"])], bm25, dense, model,
                reranker=OracleReranker("discount"), rerank_top_k=4, texts=DOCS)

    assert row["config"]["rerank"] == "oracle"
    assert row["config"]["rerank_top_k"] == 4
    assert row["per_query"][0]["rerank_ms"] == 1.5
    assert row["overall"]["p50_rerank_ms"] == 1.5
    assert row["per_query"][0]["found_at"] == 1


def test_rerank_latency_is_added_to_the_scored_latency(indices):
    bm25, dense, model = indices
    plain = H.run(H.RunConfig("thread_aware", "fake", "bm25"),
                  [_q("q1", "pricing", "semantic", ["m1"])], bm25, dense, model)
    reranked = H.run(H.RunConfig("thread_aware", "fake", "bm25", rerank="oracle"),
                     [_q("q1", "pricing", "semantic", ["m1"])], bm25, dense, model,
                     reranker=OracleReranker("discount"), rerank_top_k=4, texts=DOCS)

    assert "p50_rerank_ms" not in plain["overall"]
    assert reranked["per_query"][0]["latency_ms"] >= 1.5


def test_reranking_without_texts_fails_loudly(indices):
    # Silently reranking against missing text would score every passage as empty
    # and publish the resulting noise as a dimension-4 result.
    bm25, dense, model = indices
    with pytest.raises(ValueError, match="needs chunk texts"):
        H.run(H.RunConfig("thread_aware", "fake", "bm25", rerank="oracle"),
              [_q("q1", "pricing", "semantic", ["m1"])], bm25, dense, model,
              reranker=OracleReranker("discount"), rerank_top_k=4)


# -- dimension 5: query transformation --------------------------------------

class FakeTransformer:
    """Emits fixed texts, so fusion behaviour is checkable without an LLM."""

    def __init__(self, dense_texts, sparse_texts, kind="multi_query"):
        self.kind = kind
        self._dense, self._sparse = dense_texts, sparse_texts
        self.stats = type("S", (), {"queries": 0, "fired": 0, "degraded": 0,
                                    "calls": 0})()
        self.seen: list[tuple[str, str]] = []

    def __call__(self, query, as_of=""):
        from emailrag.query.transform import TransformedQuery
        self.seen.append((query, as_of))
        self.stats.queries += 1
        self.stats.fired += 1
        self.stats.calls += 1
        return TransformedQuery(query, self._dense, self._sparse, kind=self.kind,
                                llm_calls=1)


def test_a_transform_that_adds_the_right_terms_finds_the_message(indices):
    bm25, dense, model = indices
    # The user's words miss entirely; a rewrite lands on the corpus's wording.
    xf = FakeTransformer(["pricing"], ["pricing model discount"])
    row = H.run(H.RunConfig("thread_aware", "fake", "hybrid_rrf", transform="fake"),
                [_q("q1", "cost structure", "semantic", ["m1"])],
                bm25, dense, model, transformer=xf)

    assert row["per_query"][0]["found_at"] == 1
    assert row["transform"]["fired"] == 1
    assert row["transform"]["llm_calls"] == 1
    assert row["config"]["transform"] == "fake"


def test_the_transform_receives_the_temporal_anchor(indices):
    bm25, dense, model = indices
    xf = FakeTransformer(["pricing"], ["pricing"])
    q = EvalQuery("q1", "what was due", "temporal", ["m1"], verified=True,
                  as_of="2001-10-30")
    H.run(H.RunConfig("thread_aware", "fake", "dense", transform="fake"),
          [q], bm25, dense, model, transformer=xf)

    assert xf.seen == [("what was due", "2001-10-30")]


def test_multi_text_runs_are_fused_within_a_modality_before_across(indices, monkeypatch):
    # Four rewrites must not outvote the sparse retriever 4:1 inside a "50/50"
    # hybrid - that would make a dimension-5 change a dimension-3 change too.
    # So RRF runs twice: once over the four dense runs, once over the two
    # modalities.
    bm25, dense, model = indices
    calls: list[int] = []
    real_rrf = H.reciprocal_rank_fusion
    monkeypatch.setattr(H, "reciprocal_rank_fusion",
                        lambda runs, **kw: calls.append(len(runs)) or real_rrf(runs, **kw))

    xf = FakeTransformer(["pricing", "audit", "lunch", "pricing"], ["pricing"])
    H.run(H.RunConfig("thread_aware", "fake", "hybrid_rrf", transform="fake"),
          [_q("q1", "pricing", "semantic", ["m1"])], bm25, dense, model,
          transformer=xf)

    assert calls == [4, 2]      # four dense rewrites, then dense-vs-sparse


def test_weighted_fusion_always_sees_exactly_two_runs(indices):
    # weighted_score_fusion pairs runs with weights positionally, so a flat fuse
    # of five runs against two weights would either crash or silently reweight
    # the retrievers.
    bm25, dense, model = indices
    seen: list[tuple[int, int]] = []
    real = H.weighted_score_fusion

    def spy(runs, weights, **kw):
        seen.append((len(runs), len(weights)))
        return real(runs, weights, **kw)

    xf = FakeTransformer(["pricing", "audit", "lunch"], ["pricing", "audit"])
    H.weighted_score_fusion = spy
    try:
        H.run(H.RunConfig("thread_aware", "fake", "hybrid_weighted", transform="fake"),
              [_q("q1", "pricing", "semantic", ["m1"])], bm25, dense, model,
              transformer=xf)
    finally:
        H.weighted_score_fusion = real

    assert seen == [(2, 2)]


def test_bm25_only_ignores_the_dense_side_of_a_transform(indices):
    bm25, dense, model = indices
    xf = FakeTransformer(["something irrelevant"], ["pricing model discount"])
    row = H.run(H.RunConfig("thread_aware", "fake", "bm25", transform="fake"),
                [_q("q1", "x", "semantic", ["m1"])], bm25, dense, model,
                transformer=xf)

    assert row["per_query"][0]["found_at"] == 1


def test_cached_transform_calls_are_not_charged_as_latency(indices):
    # A cached call takes microseconds; reporting that as the transform's cost
    # would say the LLM round-trip is free.
    bm25, dense, model = indices
    xf = FakeTransformer(["pricing"], ["pricing"])
    xf._llm = type("L", (), {"last_cached": True})()

    row = H.run(H.RunConfig("thread_aware", "fake", "dense", transform="fake"),
                [_q("q1", "pricing", "semantic", ["m1"])], bm25, dense, model,
                transformer=xf)

    assert "p50_transform_ms" not in row["overall"]


def test_run_config_label_names_the_active_arms():
    cfg = H.RunConfig("thread_aware", "BAAI/bge-base-en-v1.5", "hybrid_rrf",
                      rerank="L2@20/t192", transform="hyde")
    label = cfg.label

    assert "bge-base-en-v1.5" in label
    assert "+hyde" in label and "+rerank:L2@20/t192" in label


def test_plain_run_label_stays_unchanged():
    assert H.RunConfig("thread_aware", "x/y", "bm25").label == "thread_aware | y | bm25"


# -- new tables -------------------------------------------------------------

def test_rerank_table_reports_the_gain_against_the_baseline(indices):
    bm25, dense, model = indices
    queries = [_q("q1", "lunch", "semantic", ["m1"])]
    base = H.run(H.RunConfig("thread_aware", "fake", "bm25", rerank="none"),
                 queries, bm25, dense, model)
    armed = H.run(H.RunConfig("thread_aware", "fake", "bm25", rerank="oracle"),
                  queries, bm25, dense, model,
                  reranker=OracleReranker("discount"), rerank_top_k=4, texts=DOCS)

    table = H.rerank_table([base, armed])
    lines = table.splitlines()

    assert "ΔnDCG" in lines[0]
    assert "rerank p50 ms" in lines[0]
    # The baseline row's own delta is 0.000, and the armed row shows a real gain.
    assert "+0.000" in lines[2]
    assert any("+" in line and "oracle" in line for line in lines[3:])


def test_rerank_table_omits_gains_when_no_baseline_was_run(indices):
    # Computing gains against the best row instead would make the winner look
    # free.
    bm25, dense, model = indices
    armed = H.run(H.RunConfig("thread_aware", "fake", "bm25", rerank="oracle"),
                  [_q("q1", "pricing", "semantic", ["m1"])], bm25, dense, model,
                  reranker=OracleReranker("discount"), rerank_top_k=4, texts=DOCS)

    table = H.rerank_table([armed])
    assert "—" in table.splitlines()[2]


def test_transform_table_shows_fired_and_degraded_counts(indices):
    bm25, dense, model = indices
    queries = [_q("q1", "pricing", "semantic", ["m1"])]
    base = H.run(H.RunConfig("thread_aware", "fake", "dense", transform="none"),
                 queries, bm25, dense, model)
    xf = FakeTransformer(["pricing"], ["pricing"], kind="hyde")
    armed = H.run(H.RunConfig("thread_aware", "fake", "dense", transform="hyde"),
                  queries, bm25, dense, model, transformer=xf)

    table = H.transform_table([base, armed])

    assert "fired" in table and "degraded" in table
    assert "1/1" in table
    assert "llm calls" in table


def test_per_class_table_renders(indices):
    bm25, dense, model = indices
    row = H.run(H.RunConfig("thread_aware", "fake", "bm25"),
                [_q("q1", "pricing", "semantic", ["m1"]),
                 _q("q2", "audit", "entity", ["m2"])], bm25, dense, model)

    table = H.per_class_table(row)
    assert "semantic" in table and "entity" in table
