import pytest

from repeatguard.live import ARMS, episode, messages
from repeatguard.world import ACTIONS, Task, World

TASK = Task(project="P-1-0", organization="Orchid-1-0", city="Kyoto", ask_city=True)


def stuck(prompt, allowed):
    """A model that searches for the project forever."""
    return "search_project"


def scripted(*actions):
    queue = list(actions)

    def ask(prompt, allowed):
        return queue.pop(0)

    return ask


def test_a_model_that_never_stops_searching_fails_at_the_cap():
    row = episode(TASK, stuck, "baseline")
    assert (row["success"], row["decisions"], row["tool_calls"], row["withheld"]) == (False, 8, 8, 0)


def test_withhold_runs_the_call_once_and_refuses_the_rest():
    row = episode(TASK, stuck, "withhold")
    assert (row["decisions"], row["tool_calls"], row["withheld"]) == (8, 1, 7)
    assert [s["withheld"] for s in row["steps"]] == [False] + [True] * 7


def test_the_refusal_is_visible_to_the_model_on_its_next_turn():
    seen = []

    def ask(prompt, allowed):
        seen.append(prompt[1]["content"])
        return "search_project"

    episode(TASK, ask, "withhold", cap=3)
    assert "not run: search_project already returned search(P-1-0): ['project']" in seen[2]
    assert "not run:" not in seen[1]


def test_mask_drops_the_withheld_call_from_the_action_list():
    lists = []
    answers = iter(["search_project", "search_project", "open_project"])

    def ask(prompt, allowed):
        lists.append(list(allowed))
        return next(answers)

    episode(TASK, ask, "mask", cap=3)
    assert lists[0] == lists[1] == list(ACTIONS)
    assert "search_project" not in lists[2]
    assert len(lists[2]) == len(ACTIONS) - 1


def test_withhold_never_changes_the_action_list():
    lists = []

    def ask(prompt, allowed):
        lists.append(list(allowed))
        return "search_project"

    episode(TASK, ask, "withhold", cap=4)
    assert all(item == list(ACTIONS) for item in lists)


def test_a_correct_run_succeeds_in_every_arm_and_nothing_is_withheld():
    path = ("search_project", "open_project", "search_org", "open_org", "answer")
    for arm in ARMS:
        row = episode(TASK, scripted(*path), arm)
        assert (row["success"], row["answer_given"], row["withheld"], row["tool_calls"]) == (True, "Kyoto", 0, 4)


def test_a_retry_after_an_empty_search_is_not_withheld():
    missing = Task(project="P-1-1", organization="Orchid-1-1", city="Kyoto", ask_city=True, project_primary_miss=True)
    path = ("search_project", "search_project", "search_project_alias", "open_project",
            "search_org", "open_org", "answer")
    row = episode(missing, scripted(*path), "mask")
    assert row["success"] is True
    assert row["withheld"] == 0


def test_an_unusable_model_reply_ends_the_run():
    row = episode(TASK, lambda prompt, allowed: None, "baseline")
    assert row["success"] is False
    assert row["decisions"] == 1
    assert row["steps"][0]["action"] is None


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown arm"):
        episode(TASK, stuck, "other")


def test_the_prompt_lists_history_as_the_recorded_runs_did():
    world = World(TASK)
    world.step("search_project")
    text = messages(world)[1]["content"]
    assert "Action: search_project\nObservation: search(P-1-0): ['project']" in text
    assert text.endswith("Next action:")
