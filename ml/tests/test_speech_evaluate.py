from ml.speech.evaluate import (
    DURATION_BINS,
    WER_BINS,
    bucket,
    entity_outcome,
    latency,
    leg_counts,
    subset_with_base,
    summarize,
)


def row(clip_id, tier="base", query_id="q1", wer=0.0, ref=1.0, got=1.0, entities=None,
        seconds=1.5, condition="clean", speed=1.0):
    return {"clip_id": clip_id, "tier": tier, "query_id": query_id, "wer": wer,
            "entities": entities or [], "condition": condition, "speed": speed,
            "entity_error_rate": (sum(not e["found"] for e in entities) / len(entities))
            if entities else None,
            "reference": {"ndcg_at_10": ref, "mrr": ref, "recall_at_100": ref},
            "from_transcript": {"ndcg_at_10": got, "mrr": got, "recall_at_100": got},
            "audio_seconds": seconds, "timings_ms": {"transcribe_ms": 100 * seconds,
                                                     "total_ms": 400.0},
            "legs": {"lexical": "ok", "e5": "ok", "two_tower": "timeout"}}


def test_bins() -> None:
    assert [bucket(v, WER_BINS) for v in (0, 0.1, 0.25, 0.3, 0.9)] == \
        ["0", "(0, 0.25]", "(0, 0.25]", "(0.25, 0.5]", "> 0.5"]
    assert [bucket(v, DURATION_BINS) for v in (0, 1.9, 2, 3, 8, 20)] == \
        ["< 2 s", "< 2 s", "< 2 s", "2–4 s", "4–8 s", "≥ 8 s"]


def test_summarize_reports_degradation_and_entities() -> None:
    rows = [row("a", wer=0.0, ref=0.8, got=0.8),
            row("b", wer=0.5, ref=0.9, got=0.3,
                entities=[{"found": False, "cer": 0.1}]),
            row("c", query_id="q2", wer=0.0, ref=0.6, got=0.7,
                entities=[{"found": True, "cer": 0.0}])]
    out = summarize(rows, lambda r: "all")["all"]
    assert out["clips"] == 3 and out["wer_zero_share"] == round(2 / 3, 4)
    assert out["entity_error_rate"] == 0.5
    assert out["reference_ndcg_at_10"] == round((0.8 + 0.9 + 0.6) / 3, 4)
    assert out["degradation_ndcg_at_10"] == round((0.0 + 0.6 - 0.1) / 3, 4)
    assert [entity_outcome(r) for r in rows[1:]] == ["near miss", "found"]
    assert entity_outcome(row("d", entities=[{"found": False, "cer": 0.8}])) == "missed"


def test_subset_with_base_pairs_derived_clips_with_their_clean_clip() -> None:
    rows = [row("q1-base"), row("q2-base", query_id="q2"),
            row("q1-noise", tier="condition", condition="noise_5db"),
            row("q3-accent", tier="accent", query_id="q3")]
    picked = subset_with_base(rows, "condition")
    assert [r["clip_id"] for r in picked] == ["q1-base", "q1-noise"]
    by = summarize(picked, lambda r: r["condition"])
    assert set(by) == {"clean", "noise_5db"}


def test_latency_and_legs() -> None:
    rows = [row("a", seconds=1.0), row("b", seconds=3.0), row("c", seconds=9.0)]
    out = latency(rows, lambda r: bucket(r["audio_seconds"], DURATION_BINS))
    assert list(out) == ["< 2 s", "2–4 s", "≥ 8 s"]
    assert out["≥ 8 s"]["transcribe_ms"]["p50"] == 900.0
    assert leg_counts(rows) == {"e5:ok": 3, "lexical:ok": 3, "two_tower:timeout": 3}
