import json

import pytest

from repeatguard.trace import Run, Step, canonical_args, load_rollouts, step_from_rollout


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def row(project, steps, success=True):
    return {"task": {"project": project}, "steps": steps, "success": success}


def test_usable_needs_ok_nonempty_and_known():
    assert Step("search", result="hit").usable
    assert not Step("search", result="", empty=True).usable
    assert not Step("search", result="boom", status="error").usable
    assert not Step("search", result="x", known=False).usable


def test_rollout_step_flags_errors_and_empty_results():
    hit = step_from_rollout({"action": "search_org", "observation": "search(A): ['org']"})
    miss = step_from_rollout({"action": "search_org", "observation": "search(A): []"})
    bad = step_from_rollout({"action": "open_org", "observation": "tool error: action requires a hit"})
    assert (hit.status, hit.empty) == ("ok", False)
    assert (miss.status, miss.empty) == ("ok", True)
    assert (bad.status, bad.empty) == ("error", False)


def test_unparsed_model_output_becomes_an_error_step():
    step = step_from_rollout({"action": None, "raw": "not an action"})
    assert step.call == "(unparsed)"
    assert step.status == "error"
    assert step.result == "not an action"


def test_canonical_args_ignores_key_order_and_treats_nothing_alike():
    assert canonical_args({"b": 1, "a": 2}) == canonical_args({"a": 2, "b": 1})
    assert canonical_args({"a": 1}) != canonical_args({"a": 2})
    assert canonical_args(None) == canonical_args({}) == canonical_args("") == ""


def test_load_rollouts_reads_runs_in_order(tmp_path):
    path = tmp_path / "r.jsonl"
    write_rows(path, [
        row("P-1", [{"action": "search_project", "observation": "search(P-1): ['project']"}], True),
        row("P-2", [{"action": "search_project", "observation": "search(P-2): []"}], False),
    ])
    runs = load_rollouts(path)
    assert [r.run_id for r in runs] == ["P-1", "P-2"]
    assert [r.success for r in runs] == [True, False]
    assert runs[1].steps[0].empty
    assert isinstance(runs[0], Run)


def test_load_rollouts_names_the_bad_line(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(row("P-1", [])) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: not JSON"):
        load_rollouts(path)


def test_load_rollouts_rejects_a_row_without_steps(tmp_path):
    path = tmp_path / "r.jsonl"
    write_rows(path, [{"task": {"project": "P-1"}, "success": True}])
    with pytest.raises(ValueError, match=r":1: missing or malformed"):
        load_rollouts(path)


def test_load_rollouts_rejects_a_repeated_run_id(tmp_path):
    path = tmp_path / "r.jsonl"
    write_rows(path, [row("P-1", []), row("P-1", [])])
    with pytest.raises(ValueError, match="appears twice"):
        load_rollouts(path)
