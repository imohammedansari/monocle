"""Tests for the scenario result and its failure report. Pure formatting."""
from monocle_test_tools.harness.evaluator import Verdict
from monocle_test_tools.harness.result import ScenarioResult, TurnRecord


def _failed_result():
    return ScenarioResult(
        test_name="flight",
        scenario="Book a flight from source to destination.",
        passed=False,
        turns_used=2,
        max_turns=2,
        failure_reason="max turns exhausted",
        turns=[
            TurnRecord(turn=1, tester_message="I need a flight out of San Francisco.",
                       target_response="Where would you like to fly to?",
                       verdict=Verdict(met=False, reason="no destination captured yet")),
            TurnRecord(turn=2, tester_message="Seattle, obviously.",
                       target_response="What date?",
                       verdict=Verdict(met=False, reason="no booking confirmation"),
                       param_tools_called=["destination"]),
        ],
    )


def test_report_leads_with_the_outcome_and_turn_count():
    report = _failed_result().report()
    assert report.startswith(
        "FAILED scenario 'Book a flight from source to destination.' after 2/2 turns")


def test_report_states_the_failure_reason():
    assert "reason: max turns exhausted" in _failed_result().report()


def test_report_states_the_last_verdict():
    assert "last verdict: no booking confirmation" in _failed_result().report()


def test_report_includes_every_turn():
    report = _failed_result().report()
    assert "turn 1" in report
    assert "turn 2" in report
    assert "I need a flight out of San Francisco." in report
    assert "Where would you like to fly to?" in report


def test_report_marks_which_params_the_tester_had_to_ask_for():
    assert "[called destination()]" in _failed_result().report()


def test_report_of_a_passing_result_says_passed():
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=True,
                            turns_used=1, max_turns=8,
                            turns=[TurnRecord(turn=1, tester_message="hi",
                                              target_response="booked",
                                              verdict=Verdict(met=True, reason="done"))])
    assert result.report().startswith("PASSED scenario 'Book a flight.' after 1/8 turns")


def test_report_handles_a_turn_with_no_verdict():
    """A turn that ended in a target error records no verdict."""
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=False,
                            turns_used=1, max_turns=8,
                            failure_reason="target error: boom",
                            turns=[TurnRecord(turn=1, tester_message="hi")])
    report = result.report()
    assert "target error: boom" in report
    assert "turn 1" in report


def test_report_of_a_result_with_no_turns_is_still_readable():
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=False,
                            turns_used=0, max_turns=8, failure_reason="target error: boom")
    assert "FAILED" in result.report()


# --- missing required params ----------------------------------------------

def test_report_names_required_params_that_were_never_requested():
    result = ScenarioResult(
        test_name="flight", scenario="Book a flight.", passed=False,
        turns_used=1, max_turns=8,
        failure_reason="required params never requested: date",
        missing_required_params=["date"],
        turns=[TurnRecord(turn=1, tester_message="hi", target_response="booked",
                          verdict=Verdict(met=True, reason="confirmed"))])
    lines = result.report().splitlines()
    # A dedicated line, not just the substring the failure_reason line already carries.
    assert "  never requested: date" in lines
    assert lines[0].startswith("FAILED")


def test_report_omits_the_line_when_nothing_is_missing():
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=True,
                            turns_used=1, max_turns=8,
                            turns=[TurnRecord(turn=1, tester_message="hi",
                                              target_response="booked",
                                              verdict=Verdict(met=True, reason="done"))])
    assert "never requested" not in result.report()


def test_missing_required_params_defaults_to_empty():
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=True,
                            turns_used=1, max_turns=8)
    assert result.missing_required_params == []
