"""Find calls that gave a run nothing new, and label how a failed run ended."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .trace import Run

SUCCESS = "success"
UNLABELLED = "unlabelled"
NO_CALLS = "no_calls"
EXECUTION = "execution"
RETRIEVAL = "retrieval"
MODEL_REPEAT = "model_repeat"
BUDGET = "budget"


def repeat_indices(run: Run) -> list[int]:
    """Steps that re-made a call and got back what the same call returned before.

    A retry after an error or an empty result is not a repeat: the first try gave
    the run nothing to build on. A call whose content is unknown is never judged.
    """
    latest: dict[tuple[str, str], str] = {}
    found: list[int] = []
    for index, step in enumerate(run.steps):
        if not step.usable:
            continue
        key = (step.call, step.args)
        if latest.get(key) == step.result:
            found.append(index)
        latest[key] = step.result
    return found


def attribute(run: Run) -> str:
    """The stage a run stopped at, read off its last step.

    This is a rule over the trace, not a finding of cause: it says where the run
    ended and whether it had been repeating itself, nothing more.
    """
    if run.success is None:
        return UNLABELLED
    if run.success:
        return SUCCESS
    if not run.steps:
        return NO_CALLS
    last = run.steps[-1]
    if last.status == "error":
        return EXECUTION
    if last.empty:
        return RETRIEVAL
    if repeat_indices(run):
        return MODEL_REPEAT
    return BUDGET


@dataclass(frozen=True)
class RunReport:
    run_id: str
    success: bool | None
    steps: int
    repeats: int
    cause: str


def report(run: Run) -> RunReport:
    return RunReport(
        run_id=run.run_id,
        success=run.success,
        steps=len(run.steps),
        repeats=len(repeat_indices(run)),
        cause=attribute(run),
    )


def summarize(runs: list[Run]) -> dict:
    reports = [report(run) for run in runs]
    failed = [r for r in reports if r.success is False]
    succeeded = [r for r in reports if r.success is True]
    return {
        "runs": len(reports),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "unlabelled": sum(1 for r in reports if r.success is None),
        "steps": sum(r.steps for r in reports),
        "repeated_steps": sum(r.repeats for r in reports),
        "succeeded_with_repeats": sum(1 for r in succeeded if r.repeats),
        "failed_with_repeats": sum(1 for r in failed if r.repeats),
        "failed_steps": sum(r.steps for r in failed),
        "failed_repeated_steps": sum(r.repeats for r in failed),
        "failed_causes": dict(sorted(Counter(r.cause for r in failed).items())),
    }
