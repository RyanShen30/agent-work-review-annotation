import json
import subprocess
from pathlib import Path

import pytest

from agentic_review_annotation_distilabel.adapters import MiniSWEAgentAdapter
from agentic_review_annotation_distilabel.evaluation import swebench


def benchmark_row():
    return {
        "instance_id": "owner__repo-1",
        "image": "swebench/example:latest",
        "repo": "owner/repo",
        "version": "1.0",
        "FAIL_TO_PASS": ["tests/test_fix.py::test_bug"],
        "PASS_TO_PASS": ["tests/test_old.py::test_ok"],
        "log_parser": "parse_log_pytest",
        "eval_type": "pass_and_fail",
        "eval_script": "#!/bin/bash\necho test",
        "problem_statement": "Fix the bug.",
        "patch": "gold patch",
    }


def trajectory():
    return {
        "instance_id": "owner__repo-1",
        "harness": "mini_swe_agent",
        "model": "example/model",
        "patch": "diff --git a/a.py b/a.py\n",
        "messages": [{"role": "user", "content": "Fix the bug."}],
    }


def successful_runner(command, *, cwd, **kwargs):
    run_id = command[command.index("--run_id") + 1]
    predictions_path = command[command.index("--predictions_path") + 1]
    prediction = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    artifact_dir = (
        cwd
        / "logs"
        / "run_evaluation"
        / run_id
        / prediction["model_name_or_path"]
        / prediction["instance_id"]
    )
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "test_output.txt").write_text("2 passed", encoding="utf-8")
    (artifact_dir / "run_instance.log").write_text("complete", encoding="utf-8")
    (artifact_dir / "report.json").write_text(
        json.dumps(
            {
                prediction["instance_id"]: {
                    "patch_is_None": False,
                    "patch_exists": True,
                    "patch_successfully_applied": True,
                    "resolved": True,
                    "infra_failure": False,
                    "tests_status": {
                        "FAIL_TO_PASS": {
                            "success": ["tests/test_fix.py::test_bug"],
                            "failure": [],
                        },
                        "PASS_TO_PASS": {
                            "success": ["tests/test_old.py::test_ok"],
                            "failure": [],
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return subprocess.CompletedProcess(command, 0, "official harness complete\n", "")


def test_official_evaluation_writes_inputs_and_backfills_trajectory(
    tmp_path, monkeypatch
):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory()), encoding="utf-8")
    monkeypatch.setattr(swebench, "load_benchmark", lambda path: [benchmark_row()])
    monkeypatch.setattr(swebench.importlib.metadata, "version", lambda name: "5.0.2")

    updated = swebench.evaluate_trajectories(
        [trajectory_path],
        benchmark_path=tmp_path / "benchmark.parquet",
        output_dir=tmp_path / "evaluation",
        config=swebench.EvaluationConfig(),
        command_runner=successful_runner,
    )

    assert updated == [trajectory_path.resolve()]
    saved = json.loads(trajectory_path.read_text(encoding="utf-8"))
    evaluation = saved["evaluation"]
    assert evaluation["status"] == "completed"
    assert evaluation["runner"] == "swebench.harness.run_evaluation"
    assert evaluation["resolved"] is True
    assert evaluation["official_report"]["patch_successfully_applied"] is True
    assert evaluation["per_test_results"] == [
        {
            "test": "tests/test_fix.py::test_bug",
            "category": "FAIL_TO_PASS",
            "status": "passed",
        },
        {
            "test": "tests/test_old.py::test_ok",
            "category": "PASS_TO_PASS",
            "status": "passed",
        },
    ]
    sample = MiniSWEAgentAdapter().adapt(saved)
    assert sample.evaluation["resolved"] is True
    assert sample.evaluation["runner"] == "swebench.harness.run_evaluation"

    run_dir = next((tmp_path / "evaluation").iterdir())
    prediction = json.loads((run_dir / "predictions.jsonl").read_text())
    dataset = json.loads((run_dir / "dataset.json").read_text())
    assert prediction["model_patch"] == trajectory()["patch"]
    assert dataset[0]["patch"] == "gold patch"


def test_official_evaluation_rejects_missing_report(tmp_path, monkeypatch):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory()), encoding="utf-8")
    monkeypatch.setattr(swebench, "load_benchmark", lambda path: [benchmark_row()])

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, "evaluation failed internally\n", ""
        )

    with pytest.raises(RuntimeError, match="produced no report"):
        swebench.evaluate_trajectories(
            [trajectory_path],
            benchmark_path=tmp_path / "benchmark.parquet",
            output_dir=tmp_path / "evaluation",
            config=swebench.EvaluationConfig(),
            command_runner=runner,
        )

    assert "evaluation" not in json.loads(trajectory_path.read_text())


def test_official_evaluation_records_patch_apply_failure(tmp_path, monkeypatch):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory()), encoding="utf-8")
    monkeypatch.setattr(swebench, "load_benchmark", lambda path: [benchmark_row()])
    monkeypatch.setattr(swebench.importlib.metadata, "version", lambda name: "5.0.2")

    def runner(command, *, cwd, **kwargs):
        run_id = command[command.index("--run_id") + 1]
        predictions_path = command[command.index("--predictions_path") + 1]
        prediction = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
        artifact_dir = (
            cwd
            / "logs"
            / "run_evaluation"
            / run_id
            / prediction["model_name_or_path"]
            / prediction["instance_id"]
        )
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "run_instance.log").write_text(
            ">>>>> Patch Apply Failed\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            command, 0, "official harness complete\n", ""
        )

    swebench.evaluate_trajectories(
        [trajectory_path],
        benchmark_path=tmp_path / "benchmark.parquet",
        output_dir=tmp_path / "evaluation",
        config=swebench.EvaluationConfig(),
        command_runner=runner,
    )

    evaluation = json.loads(trajectory_path.read_text())["evaluation"]
    assert evaluation["status"] == "patch_apply_failed"
    assert evaluation["resolved"] is False
    assert evaluation["official_report"]["patch_successfully_applied"] is False


def test_official_evaluation_rejects_non_swebench_dataset(tmp_path, monkeypatch):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory()), encoding="utf-8")
    row = benchmark_row()
    row.pop("eval_script")
    monkeypatch.setattr(swebench, "load_benchmark", lambda path: [row])

    with pytest.raises(ValueError, match="missing fields: eval_script"):
        swebench.evaluate_trajectories(
            [trajectory_path],
            benchmark_path=tmp_path / "benchmark.parquet",
            output_dir=tmp_path / "evaluation",
            config=swebench.EvaluationConfig(),
            command_runner=successful_runner,
        )
