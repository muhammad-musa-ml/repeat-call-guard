import json

import pytest

from repeatguard.analyze import MODEL_REPEAT, attribute, repeat_indices
from repeatguard.sim_spans import load_spans, runs_from_spans, step_from_tool_span


def tool(name, input=None, output=None, status="success", start="2026-01-01T00:00:01Z", **extra):
    span = {"id": f"t-{name}-{start}", "name": name, "type": "tool", "status": status,
            "startTime": start, "endTime": start, "duration": 1}
    if input is not None:
        span["input"] = input
    if output is not None:
        span["output"] = output
    span.update(extra)
    return span


def agent(children, status="success", span_id="agent-1"):
    return {"id": span_id, "name": "Agent", "type": "agent", "status": status, "children": children}


def test_a_tool_child_becomes_a_step_with_canonical_arguments():
    step = step_from_tool_span(tool("search", {"q": "x", "n": 1}, {"hits": ["a"]}))
    assert step.call == "search"
    assert step.args == '{"n":1,"q":"x"}'
    assert step.result == '{"hits":["a"]}'
    assert step.usable


def test_the_status_field_alone_marks_a_failed_call():
    with_text = step_from_tool_span(tool("open", output={"error": "boom"}, status="error"))
    without_text = step_from_tool_span(tool("open", output={"hits": ["a"]}, status="error"))
    assert with_text.status == without_text.status == "error"
    assert not without_text.usable


def test_an_error_key_in_a_successful_calls_output_is_data_not_a_failure():
    step = step_from_tool_span(tool("open", output={"error": None, "hits": ["a"]}))
    assert step.status == "ok" and step.usable


def test_an_empty_output_is_empty_and_a_missing_output_is_unknown():
    empty = step_from_tool_span(tool("search", output={}))
    assert empty.empty and empty.known and not empty.usable
    missing = step_from_tool_span(tool("search"))
    assert not missing.known and not missing.usable


def test_results_that_are_not_objects_arrive_wrapped_and_empty_ones_are_still_empty():
    for output in ({"value": []}, {"value": ""}, {"value": None}, {"value": {}}):
        step = step_from_tool_span(tool("search", output=output))
        assert step.empty and not step.usable, output
    for output in ({"value": [1]}, {"value": "text"}, {"value": 0}, {"value": False}):
        step = step_from_tool_span(tool("search", output=output))
        assert not step.empty and step.usable, output


def test_an_object_that_merely_has_a_value_key_among_others_is_not_unwrapped():
    step = step_from_tool_span(tool("search", output={"value": [], "hits": ["a"]}))
    assert not step.empty


def test_withheld_content_is_never_judged():
    step = step_from_tool_span(tool("search", {"q": "x"}, {"hits": ["a"]}, contentWithheld=True))
    assert not step.known


def test_an_agent_span_that_calls_the_same_tool_twice_is_flagged():
    spans = [agent([
        {"id": "m1", "name": "gpt", "type": "model"},
        tool("search", {"q": "x"}, {"hits": ["a"]}, start="2026-01-01T00:00:01Z"),
        tool("search", {"q": "x"}, {"hits": ["a"]}, start="2026-01-01T00:00:02Z"),
        tool("open", {"id": "a"}, {"text": "t"}, start="2026-01-01T00:00:03Z"),
    ], status="error")]
    (run,) = runs_from_spans(spans)
    assert [s.call for s in run.steps] == ["search", "search", "open"]
    assert repeat_indices(run) == [1]
    assert run.success is False
    assert run.run_id == "agent-1"
    assert attribute(run) == MODEL_REPEAT


def test_children_are_ordered_by_start_time():
    spans = [agent([
        tool("open", {"id": "a"}, {"text": "t"}, start="2026-01-01T00:00:03Z"),
        tool("search", {"q": "x"}, {"hits": ["a"]}, start="2026-01-01T00:00:01Z"),
    ])]
    (run,) = runs_from_spans(spans)
    assert [s.call for s in run.steps] == ["search", "open"]


def test_start_times_are_compared_as_times_not_as_text():
    later = tool("open", output={"x": 1}, start="2026-01-01T00:00:01.500Z")
    earlier = tool("search", output={"x": 1}, start="2026-01-01T00:00:01Z")
    (run,) = runs_from_spans([agent([later, earlier])])
    assert [s.call for s in run.steps] == ["search", "open"]
    offset = tool("fetch", output={"x": 1}, start="2026-01-01T02:00:00+02:00")
    plain = tool("list", output={"x": 1}, start="2026-01-01T00:30:00Z")
    (run,) = runs_from_spans([agent([plain, offset])])
    assert [s.call for s in run.steps] == ["fetch", "list"]


def test_an_unreadable_start_time_keeps_the_given_order():
    first, second = tool("b", output={"x": 1}, start="soon"), tool("a", output={"x": 1})
    (run,) = runs_from_spans([agent([first, second])])
    assert [s.call for s in run.steps] == ["b", "a"]


def test_spans_without_start_times_keep_their_order():
    first, second = tool("b", output={"x": 1}), tool("a", output={"x": 1})
    del first["startTime"], second["startTime"]
    (run,) = runs_from_spans([agent([first, second])])
    assert [s.call for s in run.steps] == ["b", "a"]


def test_every_span_holding_tools_is_its_own_run_even_when_nested():
    inner = agent([tool("search", output={"h": 1})], span_id="inner")
    outer = {"id": "loop", "name": "Loop", "type": "loop", "children": [inner, agent([tool("open", output={"t": 1})], span_id="other")]}
    runs = runs_from_spans([outer])
    assert [r.run_id for r in runs] == ["inner", "other"]


def test_spans_with_no_tool_children_give_no_run():
    assert runs_from_spans([agent([{"id": "m", "name": "gpt", "type": "model"}])]) == []
    assert runs_from_spans([]) == []


def test_block_status_maps_to_the_run_outcome():
    ok = runs_from_spans([agent([tool("a", output={"x": 1})], status="success")])[0]
    unknown = runs_from_spans([{"id": "x", "type": "agent", "children": [tool("a", output={"x": 1})]}])[0]
    assert ok.success is True and unknown.success is None


def test_load_spans_accepts_a_list_an_object_and_a_log_record(tmp_path):
    spans = [agent([tool("a", output={"x": 1})])]
    for name, body in {
        "list.json": spans,
        "obj.json": {"traceSpans": spans},
        "rec.json": {"executionData": {"traceSpans": spans}},
    }.items():
        path = tmp_path / name
        path.write_text(json.dumps(body), encoding="utf-8")
        assert len(load_spans(path)) == 1


def test_load_spans_rejects_other_shapes(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"spans": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a list of spans"):
        load_spans(path)
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError, match="not JSON"):
        load_spans(path)
