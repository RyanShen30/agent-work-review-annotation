import os
import sys

import main
from main import apply_overrides, configure_n_workers, model_env


def test_script_overrides_nested_config():
    config = {"generation": {"instance": 10}}

    apply_overrides(config, ["generation.instance=-1", "review.use_cache=false"])

    assert config == {"generation": {"instance": -1}, "review": {"use_cache": False}}


def test_n_workers_applies_to_each_phase():
    config = {}

    assert configure_n_workers(config, 4)
    assert config == {
        "n_workers": 4,
        "generation": {"n_workers": 4},
        "evaluation": {"max_workers": 4},
        "review": {"n_workers": 4},
    }


def test_model_env_does_not_alias_credentials(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")

    env = model_env()

    assert env["LLM_API_KEY"] == "secret"
    assert env["LLM_BASE_URL"] == "https://example.test/v1"
    assert "AGENTIC_REVIEW_API_KEY" not in env
    assert "AGENTIC_REVIEW_BASE_URL" not in env


def test_full_mode_reviews_trajectory_reported_by_generation(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "generation:\n  harness: mini_swe_agent\n  instance: 10\n", encoding="utf-8"
    )
    output_dir = tmp_path / "output" / "pipeline" / "traj"
    output_dir.mkdir(parents=True)
    wanted = output_dir / "wanted.json"
    wanted.write_text("{}", encoding="utf-8")
    unrelated = output_dir / "unrelated.json"
    unrelated.write_text("{}", encoding="utf-8")
    os.utime(wanted, (1, 1))
    os.utime(unrelated, (2, 2))
    commands = []
    cleaned = []

    def fake_run(command, env):
        commands.append(command)
        return str(wanted) if len(commands) == 1 else ""

    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "run", fake_run)
    monkeypatch.setattr(
        main, "cleanup_docker_images", lambda paths: cleaned.extend(paths)
    )
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--config", str(config), "--mode", "full"]
    )

    main.main()

    review_command = commands[1]
    assert review_command[review_command.index("--input") + 1] == str(wanted)
    assert cleaned == [wanted]


def test_full_mode_keeps_images_when_review_fails_with_on_success(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.yaml"
    config.write_text(
        "mode: full\ncleanup_policy: on_success\ngeneration:\n  harness: mini_swe_agent\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output" / "pipeline" / "traj"
    output_dir.mkdir(parents=True)
    trajectory = output_dir / "run.json"
    trajectory.write_text("{}", encoding="utf-8")
    calls = 0
    cleaned = []

    def fake_run(command, env):
        nonlocal calls
        calls += 1
        if calls == 1:
            return str(trajectory)
        raise SystemExit("review failed")

    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "run", fake_run)
    monkeypatch.setattr(
        main, "cleanup_docker_images", lambda paths: cleaned.extend(paths)
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config)])

    try:
        main.main()
    except SystemExit as exc:
        assert str(exc) == "review failed"
    else:
        raise AssertionError("review failure should propagate")

    assert cleaned == []


def test_full_mode_runs_official_evaluation_before_review(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "mode: full\n"
        "generation:\n  harness: mini_swe_agent\n  benchmark_path: benchmark.parquet\n"
        "evaluation:\n  enabled: true\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output" / "pipeline" / "traj"
    output_dir.mkdir(parents=True)
    trajectory = output_dir / "run.json"
    trajectory.write_text("{}", encoding="utf-8")
    commands = []

    def fake_run(command, env):
        commands.append(command)
        return str(trajectory) if len(commands) == 1 else ""

    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "run", fake_run)
    monkeypatch.setattr(main, "cleanup_docker_images", lambda paths: None)
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config)])

    main.main()

    assert commands[1][0:3] == [
        sys.executable,
        "-m",
        "agentic_review_annotation_distilabel.evaluation.swebench",
    ]
    assert commands[2][0:3] == [
        sys.executable,
        "-m",
        "agentic_review_annotation_distilabel.run",
    ]
