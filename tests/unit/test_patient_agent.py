import pytest

from cmedalign.agents.episode import run_episode
from cmedalign.agents.patient import (
    PATIENT_VISIBLE_FIELDS,
    SCORING_ONLY_FIELDS,
    PatientAgent,
    build_doctor_visible_messages,
    build_patient_visible_profile,
)
from cmedalign.data.schemas import Message, PatientProfile, RequiredInfoItem


def make_profile(**overrides) -> PatientProfile:
    base = dict(
        profile_id="p1",
        chief_complaint="chest pain",
        demographics={"age": 52, "sex": "M"},
        symptoms=["chest pain", "sweating"],
        time_course="started 1 hour ago",
        history="smoker, hypertension",
        medications=["amlodipine"],
        allergies=[],
        tests={},
        red_flags=["chest pain radiating to left arm"],
        required_info=[RequiredInfoItem(item="chest pain radiating to left arm", weight=1.0)],
        target_assessment="acute coronary syndrome",
        acceptable_actions=["call emergency services", "refer to ER immediately"],
        forbidden_claims=["this is definitely just heartburn, no need to worry"],
        source_train_id="train-001",
    )
    base.update(overrides)
    return PatientProfile(**base)


def test_patient_visible_profile_excludes_scoring_only_fields():
    profile = make_profile()
    visible = build_patient_visible_profile(profile)
    assert set(visible.keys()) == set(PATIENT_VISIBLE_FIELDS)
    for forbidden in SCORING_ONLY_FIELDS:
        assert forbidden not in visible


def test_doctor_visible_messages_never_contain_profile_object():
    history = [
        Message(role="assistant", content="How long has the pain lasted?"),
        Message(role="user", content="About an hour."),
    ]
    visible = build_doctor_visible_messages(history)
    assert visible == [
        {"role": "assistant", "content": "How long has the pain lasted?"},
        {"role": "user", "content": "About an hour."},
    ]
    for msg in visible:
        assert "target_assessment" not in msg
        assert "acute coronary syndrome" not in msg["content"]


def test_patient_leak_of_hidden_target_detected():
    profile = make_profile()

    def leaky_responder(visible_profile, doctor_question, revealed_so_far):
        return "I looked it up and I'm pretty sure I have acute coronary syndrome."

    agent = PatientAgent(profile=profile, responder=leaky_responder)
    result = agent.answer("What's wrong with you?")
    assert result.leaked_hidden_target is True


def test_patient_forbidden_claim_detected_as_leak():
    profile = make_profile()

    def responder(visible_profile, doctor_question, revealed_so_far):
        return "It's fine, this is definitely just heartburn, no need to worry."

    agent = PatientAgent(profile=profile, responder=responder)
    result = agent.answer("Are you okay?")
    assert result.leaked_hidden_target is True


def test_patient_clean_answer_not_flagged_as_leak():
    profile = make_profile()

    def responder(visible_profile, doctor_question, revealed_so_far):
        return "The pain started about an hour ago and I've been sweating a lot."

    agent = PatientAgent(profile=profile, responder=responder)
    result = agent.answer("When did it start?")
    assert result.leaked_hidden_target is False
    assert result.contradiction_detected is False


def test_patient_consistency_fixture_wrong_age_flagged():
    profile = make_profile()

    def lying_responder(visible_profile, doctor_question, revealed_so_far):
        return "I'm 30 years old and otherwise healthy."

    agent = PatientAgent(profile=profile, responder=lying_responder)
    result = agent.answer("How old are you?")
    assert result.contradiction_detected is True


def test_patient_consistency_fixture_matching_age_not_flagged():
    profile = make_profile()

    def truthful_responder(visible_profile, doctor_question, revealed_so_far):
        return "我今年52岁。"

    agent = PatientAgent(profile=profile, responder=truthful_responder)
    result = agent.answer("你多大了？")
    assert result.contradiction_detected is False


def test_patient_reveals_required_info_when_mentioned():
    profile = make_profile()

    def responder(visible_profile, doctor_question, revealed_so_far):
        return "Yes, the chest pain radiating to left arm is the main issue."

    agent = PatientAgent(profile=profile, responder=responder)
    result = agent.answer("Does the pain spread anywhere?")
    assert "chest pain radiating to left arm" in result.newly_revealed


def test_episode_terminates_on_final_marker():
    profile = make_profile()
    turns = iter(["Tell me more about your symptoms.", "FINAL: likely ACS, refer to ER"])

    def doctor_policy(visible_messages):
        return next(turns)

    def patient_responder(visible_profile, doctor_question, revealed_so_far):
        return "The pain started an hour ago."

    result = run_episode(profile, doctor_policy, patient_responder, turn_budget=8)
    assert result.terminated_reason == "final_marker"
    assert result.num_turns == 2
    assert result.valid is True


def test_episode_terminates_on_turn_budget_when_no_final_marker():
    profile = make_profile()

    def doctor_policy(visible_messages):
        return "Can you tell me more?"

    def patient_responder(visible_profile, doctor_question, revealed_so_far):
        return "It still hurts."

    result = run_episode(profile, doctor_policy, patient_responder, turn_budget=3)
    assert result.terminated_reason == "turn_budget"
    assert result.num_turns == 3


def test_episode_marked_invalid_when_patient_leaks():
    profile = make_profile()

    def doctor_policy(visible_messages):
        return "What do you think is wrong?"

    def leaky_patient_responder(visible_profile, doctor_question, revealed_so_far):
        return "The doctor said it's acute coronary syndrome."

    result = run_episode(profile, doctor_policy, leaky_patient_responder, turn_budget=2)
    assert result.valid is False
    assert any("leaked" in r for r in result.invalid_reasons)
