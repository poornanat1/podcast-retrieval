import json

from ml.monitoring import drift


def rows(n, *, zero=0, lexical_timeout=0, two_tower_error=0, rrf=0, score=3.0, total_ms=300.0,
         sources=("lexical", "e5", "two_tower"), bundle="b1"):
    out = []
    for i in range(n):
        legs = {"lexical": {"status": "timeout" if i < lexical_timeout else "ok", "count": 50},
                "e5": {"status": "ok", "count": 200},
                "two_tower": {"status": "error" if i < two_tower_error else "ok", "count": 200}}
        results = [] if i < zero else [{"score": score + 0.1 * (i % 5), "sources": list(sources)},
                                       {"score": score - 1, "sources": ["e5"]}]
        out.append({"ordering": "rrf" if i < rrf else "ranker", "bundle_id": bundle, "legs": legs,
                    "timings_ms": {"total_ms": total_ms + i, "retrieve_ms": total_ms * 0.8},
                    "results": results, "input": {"mode": "text"}})
    return out


def test_summarize_counts_everything():
    s = drift.summarize(rows(10, zero=2, lexical_timeout=5, rrf=1))
    assert s["searches"] == 10
    assert s["zero_result_rate"] == 0.2 and s["rrf_rate"] == 0.1
    assert s["leg_status"]["lexical"]["timeout"] == 0.5
    assert s["timings_ms"]["total_ms"]["n"] == 10
    assert abs(sum(s["top_score_hist"]) - 1) < 1e-9
    assert s["source_mix"]["e5"] > s["source_mix"]["lexical"]
    assert s["bundles"] == {"b1": 10} and s["inputs"] == {"text": 10}


def test_psi_is_zero_for_identical_and_grows_with_shift():
    a = drift.histogram([1.0, 2.0, 3.0] * 10, drift.SCORE_BINS)
    b = drift.histogram([7.0, 8.0, 9.0] * 10, drift.SCORE_BINS)
    assert drift.psi(a, a) == 0
    assert drift.psi(a, b) > 1


def test_compare_finds_nothing_when_stable():
    base = drift.summarize(rows(200))
    cur = drift.summarize(rows(100))
    assert [f.name for f in drift.compare(base, cur)] == []


def test_compare_flags_a_dead_leg_and_the_source_mix():
    base = drift.summarize(rows(200))
    cur = drift.summarize(rows(100, two_tower_error=100, sources=("lexical", "e5")))
    names = {f.name: f for f in drift.compare(base, cur)}
    assert names["leg_two_tower"].severity == "critical"
    assert "source_mix_psi" in names


def test_compare_flags_zero_results_latency_rrf_and_scores():
    base = drift.summarize(rows(200))
    cur = drift.summarize(rows(100, zero=40, rrf=10, score=-2.0, total_ms=1500))
    names = {f.name for f in drift.compare(base, cur)}
    assert {"zero_results", "rrf_fallback", "latency_total_ms", "top_score_psi"} <= names


def test_compare_needs_a_minimum_sample():
    base = drift.summarize(rows(200))
    findings = drift.compare(base, drift.summarize(rows(5, zero=5)))
    assert [f.severity for f in findings] == ["info"]


def test_compare_features_and_bundle_change():
    base = drift.summarize(rows(200))
    base["features"] = {"e5_sim": {"mean": 0.8, "count": 1000, "nan": 0},
                        "log_age_days": {"mean": 5.0, "count": 1000, "nan": 0}}
    cur = drift.summarize(rows(100, bundle="b2"))
    cur["features"] = {"e5_sim": {"mean": 0.3, "count": 500, "nan": 2},
                       "log_age_days": {"mean": 5.1, "count": 500, "nan": 0},
                       "new_feature": {"mean": 1, "count": 500, "nan": 0}}
    names = {f.name: f for f in drift.compare(base, cur)}
    assert names["feature_e5_sim"].severity == "warning"
    assert names["feature_nan_e5_sim"].severity == "critical"
    assert "feature_log_age_days" not in names
    assert names["bundle"].severity == "info"


def test_report_round_trips_through_json(tmp_path):
    cur = drift.summarize(rows(30))
    text = json.dumps(cur)
    assert json.loads(text)["searches"] == 30
    assert "0 finding(s)" in drift.format_report([], cur, 6)
