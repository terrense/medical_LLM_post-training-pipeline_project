import json

from cmedalign.data.schemas import Message, PatientProfile, RequiredInfoItem
from cmedalign.eval.api_adapter import EndpointConfig, OpenAICompatibleClient
from cmedalign.rewards.judge_scorer import score_episode_with_judge


def make_profile(**overrides) -> PatientProfile:
    base = dict(
        profile_id="p1",
        chief_complaint="chest pain",
        demographics={"age": 52, "sex": "M"},
        symptoms=["chest pain"],
        time_course="1 hour",
        history="smoker",
        medications=[],
        allergies=[],
        tests={},
        red_flags=["chest pain radiating to left arm"],
        required_info=[RequiredInfoItem(item="chest pain radiating to left arm", weight=1.0)],
        target_assessment="acute coronary syndrome",
        acceptable_actions=["refer to ER"],
        forbidden_claims=["this is just heartburn"],
        source_train_id="t1",
    )
    base.update(overrides)
    return PatientProfile(**base)


class FakeResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


def test_score_episode_parses_valid_judge_json(tmp_path):
    def fake_post(url, headers, json_body):
        content = json.dumps(
            {
                "clinical_correctness": 0.8,
                "safety_score": 0.9,
                "process_score": 0.7,
                "communication_score": 0.6,
                "rationale": "reasonable",
            }
        )
        return FakeResponse(200, {"model": "test-judge", "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})

    cfg = EndpointConfig(alias="TEST_JUDGE", base_url="https://example.com/v1", api_key="sk-test", model="test-judge")
    client = OpenAICompatibleClient(cfg, cache_dir=tmp_path, sleep_fn=lambda s: None, post_fn=fake_post)

    profile = make_profile()
    history = [Message(role="assistant", content="How long has the pain lasted?")]
    scores = score_episode_with_judge(client, profile, history)

    assert scores.parser_status == "ok"
    assert scores.clinical_correctness == 0.8
    assert scores.safety_score == 0.9


def test_score_episode_handles_unparseable_judge_output_gracefully(tmp_path):
    def fake_post(url, headers, json_body):
        return FakeResponse(200, {"model": "test-judge", "choices": [{"message": {"content": "not json at all"}, "finish_reason": "stop"}], "usage": {}})

    cfg = EndpointConfig(alias="TEST_JUDGE", base_url="https://example.com/v1", api_key="sk-test", model="test-judge")
    client = OpenAICompatibleClient(cfg, cache_dir=tmp_path, sleep_fn=lambda s: None, post_fn=fake_post)

    profile = make_profile()
    history = [Message(role="assistant", content="FINAL: refer to ER")]
    scores = score_episode_with_judge(client, profile, history)

    assert scores.parser_status == "unparseable"
    assert scores.clinical_correctness == 0.5  # neutral fallback, not a crash


def test_score_episode_clamps_out_of_range_values(tmp_path):
    def fake_post(url, headers, json_body):
        content = json.dumps(
            {"clinical_correctness": 1.5, "safety_score": -0.3, "process_score": 0.5, "communication_score": 0.5}
        )
        return FakeResponse(200, {"model": "test-judge", "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})

    cfg = EndpointConfig(alias="TEST_JUDGE", base_url="https://example.com/v1", api_key="sk-test", model="test-judge")
    client = OpenAICompatibleClient(cfg, cache_dir=tmp_path, sleep_fn=lambda s: None, post_fn=fake_post)

    scores = score_episode_with_judge(client, make_profile(), [Message(role="assistant", content="x")])
    assert scores.clinical_correctness == 1.0
    assert scores.safety_score == 0.0
