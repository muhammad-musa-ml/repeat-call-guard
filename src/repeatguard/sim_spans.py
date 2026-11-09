"""Read tool calls from trace spans in the shape Sim's executor writes.

The shape comes from the TraceSpan type in apps/sim/lib/logs/types.ts and the
span builder in apps/sim/lib/logs/execution/trace-spans/factory.ts of
simstudioai/sim (checked at commit 525cd52, v0.9.50). A tool call is a child
span with `type: "tool"`, its arguments in `input` and its result in `output`.
`status` is "error" when the tool or the model segment reported one; the tool's
own error text is in `output` and the segment's is in `errorMessage`. A result
that is not an object is stored as `{"value": result}`, and a tool that returned
nothing has no `output`. `contentWithheld` marks a span whose content was removed
before storage.

Legacy spans keep tool calls in `toolCalls` instead of child spans; those are not
read, so a file of them gives no runs. The spans in the tests are built to the
shape above. I have not run this on an export from a running Sim.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .trace import Run, Step, canonical_args


def _result_text(output: object) -> str:
    return output if isinstance(output, str) else canonical_args(output)


def _is_empty(output: object) -> bool:
    inner = output["value"] if isinstance(output, dict) and set(output) == {"value"} else output
    return inner is None or inner == {} or inner == [] or inner == ""


def step_from_tool_span(span: dict) -> Step:
    call = str(span.get("name") or "(unnamed)")
    args = canonical_args(span.get("input"))
    output = span.get("output")
    status = "error" if span.get("status") == "error" else "ok"
    if span.get("contentWithheld") is True or output is None:
        # Nothing to compare a repeat against, so the step is never judged.
        return Step(call, args, "", status, False, known=False)
    return Step(call, args, _result_text(output), status, empty=_is_empty(output))


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _ordered(children: list[dict]) -> list[dict]:
    """By start time when every child has a readable one, else as given."""
    times = [_parse_time(child.get("startTime")) for child in children]
    if any(t is None for t in times):
        return children
    return [child for _, child in sorted(zip(times, children), key=lambda pair: pair[0])]


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
