"""Run the retrieval world against a local model, with and without the guard.

Three arms, each with the same step cap:

baseline  the harness runs every call the model picks.
withhold  a call that repeats one with a usable result is not run; the model
          is told so in the history and loses that step.
mask      as withhold, and the withheld call is dropped from the model's
          action list for the rest of the run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import json
from urllib.request import Request, urlopen

from .guard import RepeatGuard
from .trace import step_from_rollout
from .world import ACTIONS, Task, World

MODEL = "qwen3:4b-instruct-2507-q4_K_M"
URL = "http://127.0.0.1:11434/api/chat"
CAP = 8
ARMS = ("baseline", "withhold", "mask")

SYSTEM = """You are a retrieval agent. Choose exactly one next action from this list:
search_project: search for the project named in the question.
search_project_alias: retry the project search using an alternate query.
open_project: open a project search hit to learn its organization.
search_org: search for the organization found in the project document.
search_org_alias: retry the organization search using an alternate query.
open_org: open an organization search hit to learn its city.
answer: submit the requested fact from an opened document.
Do not answer before opening the document containing the requested fact.
When a search returns a hit, open that hit before searching again.
Use an alias search only after the matching primary search returns no hits.
Do not repeat a search that has already returned a hit.
Follow these decision rules in order:
1. If Current search hit is project, choose open_project.
2. If Current search hit is org, choose open_org.
3. If the project document is not opened, choose search_project_alias after an empty primary project search; otherwise choose search_project.
4. If the question asks for the organization, choose answer after opening the project document.
5. If the organization document is not opened, choose search_org_alias after an empty primary organization search; otherwise choose search_org.
6. After opening the organization document, choose answer.
Reply with only the action name. No explanation, punctuation, or extra text."""

Ask = Callable[[list[dict[str, str]], list[str]], str | None]


def messages(world: World) -> list[dict[str, str]]:
    history = "\n".join(
        f"Action: {item['action']}\nObservation: {item['observation']}"
        for item in world.history
    ) or "No tool calls yet."
    state = (
        f"Current search hit: {world.hits[0] if world.hits else 'none'}; "
        f"project document opened: {world.opened_project}; "
        f"organization document opened: {world.opened_org}; "
        f"primary project search returned empty: {world.project_primary_empty}; "
        f"primary organization search returned empty: {world.org_primary_empty}."
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question: {world.task.question}\nHistory:\n{history}\n{state}\nNext action:"},
    ]


def ollama_ask(model: str = MODEL, url: str = URL) -> Ask:
    """Ask a local Ollama model for one action, limited to the allowed list."""

    def ask(prompt: list[dict[str, str]], allowed: list[str]) -> str | None:
        body = json.dumps({
            "model": model,
            "stream": False,
            "messages": prompt,
            "format": {
                "type": "object",
                "properties": {"action": {"type": "string", "enum": allowed}},
                "required": ["action"],
                "additionalProperties": False,
            },
            "options": {"temperature": 0, "num_predict": 64},
        }).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            raw = json.load(response)["message"]["content"]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        action = parsed.get("action") if isinstance(parsed, dict) else None
        return action if action in allowed else None

    return ask


def episode(task: Task, ask: Ask, arm: str = "baseline", cap: int = CAP) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; use one of {ARMS}")
    world = World(task)
    guard = RepeatGuard()
    masked: set[str] = set()
    steps: list[dict] = []
    for _ in range(cap):
        allowed = [a for a in ACTIONS if a not in masked]
        action = ask(messages(world), allowed)
        if action is None:
            steps.append({"action": None, "observation": "", "withheld": False})
            break
        earlier = guard.withhold(action) if arm != "baseline" else None
        if earlier is not None:
            observation = f"not run: {action} already returned {earlier}"
            world.history.append({"action": action, "observation": observation})
            if arm == "mask":
                masked.add(action)
            steps.append({"action": action, "observation": observation, "withheld": True})
            continue
        observation = world.step(action)
        guard.observe(step_from_rollout({"action": action, "observation": observation}))
        steps.append({"action": action, "observation": observation, "withheld": False})
        if world.done:
            break
    return {
        "task": asdict(task),
        "arm": arm,
        "success": world.answer_given == task.answer,
        "answer_given": world.answer_given,
        "decisions": len(steps),
        "tool_calls": world.tool_calls,
        "withheld": len(guard.withheld),
        "steps": steps,
    }
