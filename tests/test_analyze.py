from repeatguard.analyze import (
    BUDGET,
    EXECUTION,
    MODEL_REPEAT,
    NO_CALLS,
    RETRIEVAL,
    SUCCESS,
    UNKNOWN_END,
    UNLABELLED,
    attribute,
    repeat_indices,
    report,
    summarize,
)
from repeatguard.trace import Run, Step


def hit(call="search", args="", result="['org']"):
    return Step(call, args, result)


def miss(call="search", args=""):
    return Step(call, args, "[]", empty=True)


def boom(call="open", args=""):
    return Step(call, args, "tool error", status="error")


def run(*steps, success=False):
    return Run("r", tuple(steps), success)


def test_a_call_that_returns_the_same_thing_again_is_a_repeat():
    assert repeat_indices(run(hit(), hit(), hit())) == [1, 2]


def test_a_changed_result_is_not_a_repeat_and_resets_the_baseline():
    steps = run(hit(result="a"), hit(result="b"), hit(result="b"))
    assert repeat_indices(steps) == [2]


def test_different_arguments_are_different_calls():
    assert repeat_indices(run(hit(args='{"q":"x"}'), hit(args='{"q":"y"}'))) == []


def test_a_retry_after_an_empty_result_is_not_a_repeat():
    assert repeat_indices(run(miss(), miss())) == []
    assert repeat_indices(run(miss(), hit())) == []


def test_errors_are_never_repeats():
    assert repeat_indices(run(boom(), boom())) == []


def test_unknown_content_is_never_judged():
    unknown = Step("search", "", "", known=False)
    assert repeat_indices(run(unknown, unknown)) == []


def test_an_error_between_two_identical_results_does_not_erase_the_first():
    assert repeat_indices(run(hit(), boom("search"), hit())) == [2]
    assert repeat_indices(run(hit(), miss(), hit())) == [2]


def test_an_empty_run_has_no_repeats():
    assert repeat_indices(run()) == []


def test_success_and_missing_labels_come_first():
    assert attribute(run(hit(), hit(), success=True)) == SUCCESS
    assert attribute(run(hit(), success=None)) == UNLABELLED


def test_a_failed_run_with_no_calls():
    assert attribute(run()) == NO_CALLS


def test_ending_on_an_error_is_execution():
    assert attribute(run(hit(), boom())) == EXECUTION


def test_ending_on_an_empty_result_is_retrieval():
    assert attribute(run(hit(), miss("open"))) == RETRIEVAL


def test_ending_on_a_step_with_unknown_content_is_not_guessed_at():
    unknown = Step("search", "", "", known=False)
    assert attribute(run(hit(), hit(), unknown)) == UNKNOWN_END
    assert attribute(run(unknown)) == UNKNOWN_END
    assert attribute(run(hit(), unknown, Step("search", "", "", "error", known=False))) == EXECUTION


def test_repeating_without_an_error_or_empty_end_is_model_repeat():
    assert attribute(run(hit(), hit(), hit())) == MODEL_REPEAT


def test_the_last_step_wins_over_earlier_repeats():
    assert attribute(run(hit(), hit(), boom())) == EXECUTION
    assert attribute(run(hit(), hit(), miss("open"))) == RETRIEVAL


def test_a_failed_run_without_repeats_ran_out_of_budget():
    assert attribute(run(hit("a"), hit("b"), hit("c"))) == BUDGET


def test_report_counts_steps_and_repeats():
    r = report(run(hit(), hit(), hit("b")))
    assert (r.steps, r.repeats, r.cause) == (3, 1, MODEL_REPEAT)


def test_summary_counts_both_outcomes():
    runs = [
        Run("a", (hit(), hit(), hit("b")), True),
        Run("b", (hit("a"), hit("b")), True),
        Run("c", (hit(), hit(), hit()), False),
        Run("d", (hit("a"), boom()), False),
    ]
    s = summarize(runs)
    assert (s["runs"], s["succeeded"], s["failed"], s["unlabelled"]) == (4, 2, 2, 0)
    assert s["steps"] == 10
    assert s["repeated_steps"] == 3
    assert s["succeeded_with_repeats"] == 1
    assert s["failed_with_repeats"] == 1
    assert (s["failed_steps"], s["failed_repeated_steps"]) == (5, 2)
    assert s["failed_causes"] == {EXECUTION: 1, MODEL_REPEAT: 1}


def test_a_run_with_no_outcome_is_counted_apart_from_failures():
    runs = [Run("a", (hit(),), None), Run("b", (hit(), boom()), False), Run("c", (hit(),), True)]
    s = summarize(runs)
    assert (s["succeeded"], s["failed"], s["unlabelled"]) == (1, 1, 1)
    assert s["failed_steps"] == 2
