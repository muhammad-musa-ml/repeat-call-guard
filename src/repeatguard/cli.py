"""Command line: `repeatguard analyze FILE`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .analyze import report, summarize
from .sim_spans import load_spans
from .trace import load_rollouts


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def format_summary(path: str, s: dict) -> str:
    lines = [
        f"file: {path}",
        f"runs: {s['runs']} (succeeded {s['succeeded']}, failed {s['failed']}, unlabelled {s['unlabelled']})",
        f"steps: {s['steps']}; repeated steps: {s['repeated_steps']} ({percent(s['repeated_steps'], s['steps'])})",
        f"succeeded runs with a repeated step: {s['succeeded_with_repeats']} of {s['succeeded']}",
        f"failed runs with a repeated step: {s['failed_with_repeats']} of {s['failed']}",
        f"failed runs: {s['failed_steps']} steps, {s['failed_repeated_steps']} of them repeated"
        f" ({percent(s['failed_repeated_steps'], s['failed_steps'])})",
        "how the failed runs ended: "
        + (", ".join(f"{cause} {count}" for cause, count in s["failed_causes"].items()) or "none"),
    ]
    return "\n".join(lines)


def format_runs(runs: list) -> str:
    lines = ["run_id  success  steps  repeats  ended_as"]
    for run in runs:
        r = report(run)
        lines.append(f"{r.run_id}  {r.success}  {r.steps}  {r.repeats}  {r.cause}")
    return "\n".join(lines)


def cmd_analyze(args: argparse.Namespace) -> int:
    runs = load_spans(args.file) if args.spans else load_rollouts(args.file)
    summary = summarize(runs)
    text = json.dumps(summary, indent=2, sort_keys=True) if args.json else format_summary(args.file, summary)
    if args.runs:
        text += "\n\n" + format_runs(runs)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    else:
        print(text)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .live import ollama_ask, episode
    from .world import make_tasks

    ask = ollama_ask(args.model, args.url)
    tasks = make_tasks(args.tasks, seed=args.seed, miss_rate=args.miss_rate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wins = {arm: 0 for arm in args.arms}
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(tasks, 1):
            for arm in args.arms:
                row = episode(task, ask, arm)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                wins[arm] += int(row["success"])
            print(f"{index}/{len(tasks)} " + " ".join(f"{a}={wins[a]}" for a in args.arms), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="repeatguard")
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("live", help="run the retrieval world against a local Ollama model")
    live.add_argument("--seed", type=int, required=True)
    live.add_argument("--tasks", type=int, default=80)
    live.add_argument("--miss-rate", type=float, default=0.25)
    live.add_argument("--arms", nargs="+", default=["baseline", "withhold", "mask"],
                      choices=["baseline", "withhold", "mask"])
    live.add_argument("--model", default="qwen3:4b-instruct-2507-q4_K_M")
    live.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    live.add_argument("--out", required=True, help="JSON Lines file, one row per task and arm")
    live.set_defaults(func=cmd_live)
    analyze = commands.add_parser("analyze", help="count repeated calls in recorded rollouts")
    analyze.add_argument("file")
    analyze.add_argument("--spans", action="store_true", help="FILE holds Sim trace spans, not rollouts")
    analyze.add_argument("--json", action="store_true", help="print the summary as JSON")
    analyze.add_argument("--runs", action="store_true", help="also list every run")
    analyze.add_argument("--out", help="write the text to this file instead of printing it")
    analyze.set_defaults(func=cmd_analyze)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError) as exc:
        print(f"repeatguard: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
