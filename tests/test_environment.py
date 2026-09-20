from agentic_review_annotation_distilabel.environment import (
    load_environment,
    model_credentials,
)


def test_scoped_credentials_use_separate_providers(monkeypatch):
    monkeypatch.setenv("GENERATION_LLM_API_KEY", "generation-key")
    monkeypatch.setenv("GENERATION_LLM_BASE_URL", "https://generation.test/v1")
    monkeypatch.setenv("REVIEW_LLM_API_KEY", "review-key")
    monkeypatch.setenv("REVIEW_LLM_BASE_URL", "https://review.test/v1")

    assert model_credentials("GENERATION") == (
        "generation-key",
        "https://generation.test/v1",
    )
    assert model_credentials("REVIEW") == ("review-key", "https://review.test/v1")


def test_scoped_credentials_fall_back_to_legacy_variables(monkeypatch):
    monkeypatch.delenv("GENERATION_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENERATION_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "legacy-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://legacy.test/v1")

    assert model_credentials("GENERATION") == (
        "legacy-key",
        "https://legacy.test/v1",
    )


def test_environment_file_does_not_override_explicit_shell_values(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GENERATION_LLM_API_KEY=file-key\n"
        "GENERATION_LLM_BASE_URL=https://file.test/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GENERATION_LLM_API_KEY", "shell-key")
    monkeypatch.delenv("GENERATION_LLM_BASE_URL", raising=False)

    assert load_environment(env_file)
    assert model_credentials("GENERATION") == (
        "shell-key",
        "https://file.test/v1",
    )
