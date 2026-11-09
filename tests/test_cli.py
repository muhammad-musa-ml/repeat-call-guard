import json
from pathlib import Path

from repeatguard.cli import main

DATA = Path(__file__).resolve().parent.parent / "data" / "llm_teacher_rollouts.jsonl"


def two_run_file(tmp_path):
    rows = [
        {"task": {"project": "P-1"}, "success": True, "steps": [
            {"action": "search_org", "observation": "search(A): ['org']"},
            {"action": "search_org", "observation": "search(A): ['org']"},
        ]},
        {"task": {"project": "P-2"}, "success": False, "steps": [
            {"action": "search_org", "observation": "search(A): []"},
        ]},
    ]
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_analyze_prints_the_counts(tmp_path, capsys):
    assert main(["analyze", str(two_run_file(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "runs: 2 (succeeded 1, failed 1, unlabelled 0)" in out
    assert "steps: 3; repeated steps: 1 (33.3%)" in out
    assert "how the failed runs ended: retrieval 1" in out


def test_analyze_json_and_run_table(tmp_path, capsys):
    assert main(["analyze", str(two_run_file(tmp_path)), "--json", "--runs"]) == 0
    out = capsys.readouterr().out
    body, table = out.split("\n\n")
    assert json.loads(body)["repeated_steps"] == 1
    assert "P-1  True  2  1  success" in table
    assert "P-2  False  1  0  retrieval" in table


def test_out_writes_the_text_to_a_file_with_lf_endings(tmp_path, capsys):
    target = tmp_path / "out.txt"
    assert main(["analyze", str(two_run_file(tmp_path)), "--out", str(target)]) == 0
    assert capsys.readouterr().out == ""
    data = target.read_bytes()
    assert data.startswith(b"file: ") and data.endswith(b"retrieval 1\n")
    assert b"\r" not in data


def test_live_writes_one_row_per_task_and_arm(tmp_path, monkeypatch, capsys):
    import repeatguard.live as live

    monkeypatch.setattr(live, "ollama_ask", lambda model, url: lambda prompt, allowed: "search_project")
    target = tmp_path / "live.jsonl"
    assert main(["live", "--seed", "5", "--tasks", "2", "--out", str(target)]) == 0
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [(r["task"]["project"], r["arm"]) for r in rows] == [
        ("P-5-0", "baseline"), ("P-5-0", "withhold"), ("P-5-0", "mask"),
        ("P-5-1", "baseline"), ("P-5-1", "withhold"), ("P-5-1", "mask"),
    ]
    assert [r["withheld"] for r in rows[:3]] == [0, 7, 7]
    assert "2/2 baseline=0 withhold=0 mask=0" in capsys.readouterr().out


def test_live_can_run_one_arm(tmp_path, monkeypatch):
    import repeatguard.live as live

    monkeypatch.setattr(live, "ollama_ask", lambda model, url: lambda prompt, allowed: "search_project")
    target = tmp_path / "live.jsonl"
    assert main(["live", "--seed", "5", "--tasks", "1", "--arms", "mask", "--out", str(target)]) == 0
    assert len(target.read_text(encoding="utf-8").splitlines()) == 1


def test_a_missing_file_exits_2(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "none.jsonl")]) == 2
    assert "repeatguard:" in capsys.readouterr().err


def test_a_bad_line_exits_2_and_names_it(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    path.write_text("{oops\n", encoding="utf-8")
    assert main(["analyze", str(path)]) == 2
    assert ":1: not JSON" in capsys.readouterr().err


def test_the_recorded_rollouts_hold_the_counts_the_readme_quotes(capsys):
    assert main(["analyze", str(DATA), "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert (s["runs"], s["succeeded"], s["failed"]) == (80, 57, 23)
