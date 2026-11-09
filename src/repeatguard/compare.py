"""Compare the arms of live runs on the same tasks."""

from __future__ import annotations

from collections import defaultdict
import json
from math import comb
from pathlib import Path

from .live import ARMS
from .trace import Run

REQUIRED = ("arm", "success", "tool_calls", "decisions", "withheld", "steps")


def _check_row(row: dict) -> None:
    for field in REQUIRED:
        if field not in row:
            raise KeyError(field)
    if not isinstance(row["task"]["project"], str):
        raise TypeError("task.project is not a string")
    if not isinstance(row["steps"], list):
        raise TypeError("steps is not a list")


def load_live(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                _check_row(row)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not JSON ({exc.msg})") from exc
            except (KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: missing or malformed field ({exc!r})") from exc
            if row["arm"] not in ARMS:
                raise ValueError(f"{path}:{number}: unknown arm {row['arm']!r}")
            rows.append(row)
    return rows


def by_arm(rows: list[dict]) -> dict[str, dict[str, dict]]:
    arms: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        project = row["task"]["project"]
        if project in arms[row["arm"]]:
            raise ValueError(f"task {project} appears twice in arm {row['arm']}")
        arms[row["arm"]][project] = row
    return dict(arms)


def sign_test_p(only_candidate: int, only_baseline: int) -> float:
    """Two-sided exact sign test on the tasks where the two arms disagree."""
    n = only_candidate + only_baseline
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(only_candidate, only_baseline) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(baseline: dict[str, dict], candidate: dict[str, dict]) -> dict:
    tasks = sorted(set(baseline) & set(candidate))
    both = sum(1 for t in tasks if baseline[t]["success"] and candidate[t]["success"])
    only_candidate = sum(1 for t in tasks if candidate[t]["success"] and not baseline[t]["success"])
    only_baseline = sum(1 for t in tasks if baseline[t]["success"] and not candidate[t]["success"])
    return {
        "tasks": len(tasks),
        "both": both,
        "only_candidate": only_candidate,
        "only_baseline": only_baseline,
        "neither": len(tasks) - both - only_candidate - only_baseline,
        "p": sign_test_p(only_candidate, only_baseline),
    }


def arm_summary(rows: dict[str, dict]) -> dict:
    n = len(rows)
    values = list(rows.values())
    return {
        "tasks": n,
        "succeeded": sum(1 for r in values if r["success"]),
        "tool_calls": sum(r["tool_calls"] for r in values) / n if n else 0.0,
        "decisions": sum(r["decisions"] for r in values) / n if n else 0.0,
        "withheld": sum(r["withheld"] for r in values) / n if n else 0.0,
    }


def recorded_agreement(baseline: dict[str, dict], recorded: list[Run]) -> tuple[int, int]:
    """Tasks whose live baseline picked the same actions, step for step, as the recorded run."""
    same = total = 0
    for run in recorded:
        row = baseline.get(run.run_id)
        if row is None:
            continue
        total += 1
        live = [s["action"] or "(unparsed)" for s in row["steps"]]
        same += int(live == [s.call for s in run.steps])
    return same, total


def common_tasks(arms: dict[str, dict[str, dict]]) -> tuple[dict[str, dict[str, dict]], int]:
    """Keep only the tasks every arm has finished, and count the rows dropped."""
    if not arms:
        return {}, 0
    shared = set.intersection(*(set(rows) for rows in arms.values()))
    dropped = sum(len(rows) for rows in arms.values()) - len(shared) * len(arms)
    return {arm: {t: rows[t] for t in shared} for arm, rows in arms.items()}, dropped


def format_report(name: str, rows: list[dict], recorded: list[Run] | None = None) -> str:
    arms, dropped = common_tasks(by_arm(rows))
    lines = [f"{name}: {len(rows)} rows"]
    if dropped:
        lines.append(f"{dropped} rows ignored because their task is missing from another arm")
    if not any(arms.values()):
        lines.append("no task has a row in every arm")
        return "\n".join(lines)
    lines.append("arm       tasks  succeeded  rate    tool calls  decisions  withheld")
    for arm in ARMS:
        if arm not in arms:
            continue
        s = arm_summary(arms[arm])
        rate = f"{100 * s['succeeded'] / s['tasks']:.1f}%"
        lines.append(
            f"{arm:<9} {s['tasks']:>5}  {s['succeeded']:>9}  {rate:>6}  {s['tool_calls']:>10.2f}"
            f"  {s['decisions']:>9.2f}  {s['withheld']:>8.2f}"
        )
    if "baseline" in arms:
        for arm in ARMS:
            if arm == "baseline" or arm not in arms:
                continue
            p = paired(arms["baseline"], arms[arm])
            lines.append(
                f"{arm} against baseline, same {p['tasks']} tasks: both {p['both']}, only {arm} {p['only_candidate']},"
                f" only baseline {p['only_baseline']}, neither {p['neither']}; exact sign test p = {p['p']:.4g}"
            )
        if recorded is not None:
            same, total = recorded_agreement(arms["baseline"], recorded)
            lines.append(f"live baseline against the recorded rollouts: {same} of {total} tasks picked the same actions at every step")
    return "\n".join(lines)
