from main import apply_overrides, model_env


def test_script_overrides_nested_config():
    config = {"generation": {"instance": 10}}

    apply_overrides(config, ["generation.instance=-1", "review.use_cache=false"])

    assert config == {"generation": {"instance": -1}, "review": {"use_cache": False}}


def test_model_env_does_not_alias_credentials(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")

    env = model_env()

    assert env["LLM_API_KEY"] == "secret"
    assert env["LLM_BASE_URL"] == "https://example.test/v1"
    assert "AGENTIC_REVIEW_API_KEY" not in env
    assert "AGENTIC_REVIEW_BASE_URL" not in env
