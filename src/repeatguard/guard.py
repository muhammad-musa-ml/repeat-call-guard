"""Withhold a tool call that repeats one the run already has an answer to."""

from __future__ import annotations

from .trace import Step


class RepeatGuard:
    """Remembers what each call last returned and refuses to run it again.

    Only a usable result arms the guard. After an error or an empty result the
    same call may run again, because that is what a retry is for.
    """

    def __init__(self) -> None:
        self._latest: dict[tuple[str, str], str] = {}
        self.withheld: list[tuple[str, str]] = []

    def observe(self, step: Step) -> None:
        """Record a step the harness actually ran."""
        if step.usable:
            self._latest[(step.call, step.args)] = step.result

    def withhold(self, call: str, args: str = "") -> str | None:
        """The earlier result when this call should not run, else None."""
        earlier = self._latest.get((call, args))
        if earlier is not None:
            self.withheld.append((call, args))
        return earlier

    def withheld_calls(self) -> frozenset[str]:
        """Names of the calls withheld so far in this run."""
        return frozenset(call for call, _ in self.withheld)
