"""Command line: `repeatguard analyze FILE`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .analyze import report, summarize
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
    runs = load_rollouts(args.file)
    summary = summarize(runs)
    text = json.dumps(summary, indent=2, sort_keys=True) if args.json else format_summary(args.file, summary)
    if args.runs:
        text += "\n\n" + format_runs(runs)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="repeatguard")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="count repeated calls in recorded rollouts")
    analyze.add_argument("file")
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
