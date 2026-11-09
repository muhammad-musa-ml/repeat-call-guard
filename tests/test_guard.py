from repeatguard.guard import RepeatGuard
from repeatguard.trace import Step


def test_a_call_is_withheld_only_after_it_returned_something_usable():
    guard = RepeatGuard()
    assert guard.withhold("search") is None
    guard.observe(Step("search", result="['org']"))
    assert guard.withhold("search") == "['org']"


def test_an_empty_result_does_not_arm_the_guard():
    guard = RepeatGuard()
    guard.observe(Step("search", result="[]", empty=True))
    assert guard.withhold("search") is None


def test_an_error_does_not_arm_the_guard():
    guard = RepeatGuard()
    guard.observe(Step("open", result="tool error", status="error"))
    assert guard.withhold("open") is None


def test_unknown_content_does_not_arm_the_guard():
    guard = RepeatGuard()
    guard.observe(Step("search", result="", known=False))
    assert guard.withhold("search") is None


def test_other_arguments_and_other_calls_still_run():
    guard = RepeatGuard()
    guard.observe(Step("search", '{"q":"a"}', "hit"))
    assert guard.withhold("search", '{"q":"b"}') is None
    assert guard.withhold("fetch", '{"q":"a"}') is None
    assert guard.withhold("search", '{"q":"a"}') == "hit"


def test_the_latest_result_is_the_one_returned():
    guard = RepeatGuard()
    guard.observe(Step("status", result="running"))
    guard.observe(Step("status", result="done"))
    assert guard.withhold("status") == "done"


def test_each_withheld_call_is_recorded():
    guard = RepeatGuard()
    guard.observe(Step("search", result="hit"))
    guard.withhold("search")
    guard.withhold("search")
    guard.withhold("open")
    assert guard.withheld == [("search", ""), ("search", "")]
