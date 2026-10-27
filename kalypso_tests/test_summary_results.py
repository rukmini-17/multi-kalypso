from types import SimpleNamespace as NS

from vllm.kalypso.benchmark.summarize_requests import summarize


def test_summary_includes_all_failed_worker():
    rows = [
        {"worker": "w0", "status": "ok", "submit_t": 1, "finish_t": 3,
         "prompt_tokens": 10, "completion_tokens": 2},
        {"worker": "w1", "status": "error", "submit_t": 1, "finish_t": 2},
        {"worker": "w1", "status": "cancelled", "submit_t": 2, "finish_t": 3},
    ]
    stats = summarize(rows, [{"worker": "w1", "waiting": 2, "running": 1}])
    assert stats == {
        "w0": {"requests": 1, "errors": 0, "lat_p50_s": 2, "lat_p99_s": 2,
               "lat_mean_s": 2, "prompt_tokens": 10, "completion_tokens": 2,
               "throughput_rps": 0.5, "peak_queue_depth": None},
        "w1": {"requests": 0, "errors": 2, "lat_p50_s": 0, "lat_p99_s": 0,
               "lat_mean_s": 0, "prompt_tokens": 0, "completion_tokens": 0,
               "throughput_rps": 0, "peak_queue_depth": 3},
    }
    failed = summarize(rows[1:], [])
    assert failed["w1"] == {**stats["w1"], "peak_queue_depth": None}
    assert summarize([], []) == {}


def test_results_helper_preserves_shape_and_unknown_outputs():
    from vllm.kalypso.results import format_results

    ctxs = [NS(
        state=NS(idx=3, worker=NS(name="gpu0")),
        input=NS(data=[{"role": "user", "content": "alpha"}], right_data=["beta"]),
        output=[{"SemMap": "mapped"}, {"a": 1, "b": 2}, "raw", {}],
    ), NS(state=NS(), input=NS(), output=[])]
    assert format_results(ctxs) == [{
        "idx": 3, "worker": "gpu0",
        "input_context": [{"role": "user", "content": "alpha"}],
        "right_input_context": ["beta"],
        "outputs": [{"op": "SemMap", "value": "mapped"},
                    {"op": "unknown", "value": {"a": 1, "b": 2}},
                    {"op": "unknown", "value": "raw"}, {"op": "unknown", "value": {}}],
    }, {"idx": None, "worker": None, "input_context": None,
        "right_input_context": None, "outputs": []}]
    assert format_results([]) == []
