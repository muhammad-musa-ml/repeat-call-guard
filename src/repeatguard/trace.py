"""A run is an ordered list of tool calls and what each one returned."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Step:
    call: str
    args: str = ""
    result: str = ""
    status: str = "ok"
    empty: bool = False
    known: bool = True

    @property
    def usable(self) -> bool:
        """The call returned something the run can build on."""
        return self.known and self.status == "ok" and not self.empty


@dataclass(frozen=True)
class Run:
    run_id: str
    steps: tuple[Step, ...]
    success: bool | None = None


ERROR_PREFIX = "tool error"
EMPTY_SUFFIX = ": []"


def canonical_args(value: object) -> str:
    """One string per distinct argument set, whatever the key order."""
    if value is None or value == {} or value == [] or value == "":
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def step_from_rollout(entry: dict) -> Step:
    """One step of a recorded rollout: an action name and a text observation."""
    action = entry.get("action")
    if action is None:
        return Step(call="(unparsed)", result=str(entry.get("raw", "")), status="error")
    observation = entry.get("observation", "")
    return Step(
        call=action,
        result=observation,
        status="error" if observation.startswith(ERROR_PREFIX) else "ok",
        empty=observation.endswith(EMPTY_SUFFIX),
    )


def load_rollouts(path: str | Path) -> list[Run]:
    """Read a JSON Lines file of rollouts with `task`, `steps` and `success`."""
    runs: list[Run] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not JSON ({exc.msg})") from exc
            try:
                run_id = row["task"]["project"]
                steps = tuple(step_from_rollout(entry) for entry in row["steps"])
                success = row["success"]
            except (KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"{path}:{number}: missing or malformed field ({exc!r})") from exc
            if success is not None and not isinstance(success, bool):
                raise ValueError(f"{path}:{number}: success must be true, false or null, got {success!r}")
            if run_id in seen:
                raise ValueError(f"{path}:{number}: run id {run_id!r} appears twice")
            seen.add(run_id)
            runs.append(Run(run_id=run_id, steps=steps, success=success))
    return runs
