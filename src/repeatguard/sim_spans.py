"""Read tool calls from trace spans in the shape Sim's executor writes.

The shape comes from the TraceSpan type in apps/sim/lib/logs/types.ts of
simstudioai/sim (checked at commit 525cd52, v0.9.50): a tool call is a child
span with `type: "tool"`, its arguments in `input` and its result in `output`,
and a failed call has `status: "error"` with the error inside `output`.
`contentWithheld` marks a span whose content was removed before storage.
The spans in the tests are built to that shape. I have not run this on an
export from a running Sim.
"""

from __future__ import annotations

import json
from pathlib import Path

from .trace import Run, Step, canonical_args


def _result_text(output: object) -> str:
    return output if isinstance(output, str) else canonical_args(output)


def step_from_tool_span(span: dict) -> Step:
    call = str(span.get("name") or "(unnamed)")
    args = canonical_args(span.get("input"))
    output = span.get("output")
    withheld = span.get("contentWithheld") is True
    failed = span.get("status") == "error" or (isinstance(output, dict) and "error" in output)
    if withheld or output is None:
        # Nothing to compare a repeat against, so the step is never judged.
        return Step(call, args, "", "error" if failed else "ok", False, known=False)
    return Step(
        call,
        args,
        _result_text(output),
        "error" if failed else "ok",
        empty=output == {} or output == [] or output == "",
    )


def _ordered(children: list[dict]) -> list[dict]:
    if all(isinstance(child.get("startTime"), str) for child in children):
        return sorted(children, key=lambda child: child["startTime"])
    return children


def _outcome(span: dict) -> bool | None:
    status = span.get("status")
    return {"success": True, "error": False}.get(status)


def runs_from_spans(spans: list[dict]) -> list[Run]:
    """One run per span that directly holds tool-call children."""
    runs: list[Run] = []

    def walk(span: dict) -> None:
        children = [c for c in span.get("children") or [] if isinstance(c, dict)]
        tools = [c for c in children if c.get("type") == "tool"]
        if tools:
            run_id = str(span.get("id") or span.get("name") or f"span-{len(runs)}")
            runs.append(Run(run_id, tuple(step_from_tool_span(t) for t in _ordered(tools)), _outcome(span)))
        for child in children:
            walk(child)

    for span in spans:
        if isinstance(span, dict):
            walk(span)
    return runs


def load_spans(path: str | Path) -> list[Run]:
    """Read a JSON file holding a span list, `{"traceSpans": [...]}` or a log record."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not JSON ({exc.msg})") from exc
    if isinstance(data, dict) and isinstance(data.get("executionData"), dict):
        data = data["executionData"]
    if isinstance(data, dict):
        data = data.get("traceSpans")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of spans, or an object with traceSpans")
    return runs_from_spans(data)
