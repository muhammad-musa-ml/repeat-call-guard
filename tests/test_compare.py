import json

import pytest

from repeatguard.compare import (
    arm_summary,
    by_arm,
    format_report,
    load_live,
    paired,
    recorded_agreement,
    sign_test_p,
)
from repeatguard.trace import Run, Step


def row(project, arm, success, tool_calls=4, decisions=5, withheld=0, actions=("a",)):
    return {
        "task": {"project": project}, "arm": arm, "success": success,
        "tool_calls": tool_calls, "decisions": decisions, "withheld": withheld,
        "steps": [{"action": a, "observation": "", "withheld": False} for a in actions],
    }


def test_sign_test_matches_hand_computed_values():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(3, 3) == 1.0
    assert sign_test_p(5, 0) == pytest.approx(2 / 32)
    assert sign_test_p(0, 5) == pytest.approx(2 / 32)
    assert sign_test_p(1, 0) == 1.0
    assert sign_test_p(10, 2) == pytest.approx(2 * (1 + 12 + 66) / 4096)


def test_paired_counts_the_four_cells():
    base = {"t1": row("t1", "baseline", True), "t2": row("t2", "baseline", False),
            "t3": row("t3", "baseline", False), "t4": row("t4", "baseline", True)}
    cand = {"t1": row("t1", "mask", True), "t2": row("t2", "mask", True),
            "t3": row("t3", "mask", False), "t4": row("t4", "mask", False)}
    p = paired(base, cand)
    assert (p["tasks"], p["both"], p["only_candidate"], p["only_baseline"], p["neither"]) == (4, 1, 1, 1, 1)


def test_paired_ignores_tasks_missing_from_one_arm():
    base = {"t1": row("t1", "baseline", True), "t2": row("t2", "baseline", True)}
    cand = {"t1": row("t1", "mask", True)}
    assert paired(base, cand)["tasks"] == 1


def test_arm_summary_averages_per_task():
    rows = {"t1": row("t1", "mask", True, 2, 4, 1), "t2": row("t2", "mask", False, 4, 8, 3)}
    s = arm_summary(rows)
    assert (s["tasks"], s["succeeded"], s["tool_calls"], s["decisions"], s["withheld"]) == (2, 1, 3.0, 6.0, 2.0)


def test_by_arm_refuses_a_task_twice_in_one_arm():
    with pytest.raises(ValueError, match="appears twice"):
        by_arm([row("t1", "mask", True), row("t1", "mask", False)])


def test_load_live_names_bad_lines_and_unknown_arms(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_text(json.dumps(row("t1", "mask", True)) + "\n{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: not JSON"):
        load_live(path)
    path.write_text(json.dumps(row("t1", "other", True)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":1: unknown arm"):
        load_live(path)
    path.write_text(json.dumps({"task": {"project": "t1"}, "arm": "mask"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":1: missing or malformed"):
        load_live(path)


def test_recorded_agreement_compares_action_sequences():
    baseline = {"t1": row("t1", "baseline", True, actions=("a", "b")),
                "t2": row("t2", "baseline", True, actions=("a", "c"))}
    recorded = [Run("t1", (Step("a"), Step("b")), True), Run("t2", (Step("a"), Step("b")), True),
                Run("t3", (Step("a"),), True)]
    assert recorded_agreement(baseline, recorded) == (1, 2)


def test_the_report_has_the_arms_the_pairing_and_the_recorded_check():
    rows = [row("t1", "baseline", False, actions=("a",)), row("t1", "mask", True)]
    recorded = [Run("t1", (Step("a"),), False)]
    text = format_report("set", rows, recorded)
    assert "mask" in text and "baseline" in text
    assert "mask against baseline, same 1 tasks: both 0, only mask 1, only baseline 0, neither 0" in text
    assert "1 of 1 tasks picked the same actions at every step" in text
