import pytest
from pydantic import ValidationError

from cmedalign.data.schemas import ConversationRecord, Message, PatientProfile, RequiredInfoItem


def _base_record(**overrides):
    base = dict(
        sample_id="s1",
        source="demo",
        source_revision="rev1",
        split="train",
        messages=[
            {"role": "system", "content": "you are a doctor"},
            {"role": "user", "content": "I have a headache"},
            {"role": "assistant", "content": "How long has it lasted?"},
        ],
        provenance="unit-test",
        license_id="cc-by-4.0",
        raw_hash="a" * 64,
        normalized_hash="b" * 64,
    )
    base.update(overrides)
    return base


def test_valid_record_parses():
    rec = ConversationRecord(**_base_record())
    assert rec.split == "train"
    assert len(rec.messages) == 3


def test_no_empty_messages():
    with pytest.raises(ValidationError):
        ConversationRecord(**_base_record(messages=[{"role": "user", "content": "   "}]))


def test_no_adjacent_same_role():
    with pytest.raises(ValidationError):
        ConversationRecord(
            **_base_record(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": "hello again"},
                ]
            )
        )


def test_unknown_role_rejected():
    with pytest.raises(ValidationError):
        ConversationRecord(**_base_record(messages=[{"role": "doctor", "content": "hi"}]))


def test_empty_messages_list_rejected():
    with pytest.raises(ValidationError):
        ConversationRecord(**_base_record(messages=[]))


def test_patient_profile_roundtrip():
    profile = PatientProfile(
        profile_id="p1",
        chief_complaint="chest pain",
        demographics={"age": 45, "sex": "F"},
        symptoms=["chest pain", "shortness of breath"],
        time_course="started 2 hours ago",
        history="hypertension",
        medications=["amlodipine"],
        allergies=[],
        tests={},
        red_flags=["chest pain radiating to arm"],
        required_info=[RequiredInfoItem(item="onset", weight=1.0)],
        target_assessment="possible ACS, needs urgent referral",
        acceptable_actions=["refer to ER"],
        forbidden_claims=["this is definitely not a heart attack"],
        source_train_id="train-123",
    )
    assert profile.required_info[0].weight == 1.0
