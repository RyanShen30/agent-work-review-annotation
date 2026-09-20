import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

from agentic_review_annotation_distilabel import datasets
from agentic_review_annotation_distilabel.agents import run
from agentic_review_annotation_distilabel.agents.base import AgentConfig, trajectory_filename
from agentic_review_annotation_distilabel.agents.mini_swe_agent import (
    MiniSWEAgent,
    _container_platform,
    _docker_run_args,
)
from agentic_review_annotation_distilabel.agents.opencollab import OpenCollabAgent
from agentic_review_annotation_distilabel.datasets import (
    load_instance,
    load_instances,
    swebench_image,
)


def test_minus_one_loads_entire_benchmark(monkeypatch):
    rows = [{"instance_id": "one"}, {"instance_id": "two"}]
    monkeypatch.setattr(datasets, "load_benchmark", lambda path: rows)

    assert load_instances("benchmark", -1) == rows


def test_existing_matching_trajectory_skips_generation(tmp_path, monkeypatch, capsys):
    problem = "Fix it."
    row = {"instance_id": "task-1", "problem_statement": problem}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "generation:\n  harness: mini_swe_agent\n  model: test/model\n  benchmark_path: unused\n",
        encoding="utf-8",
    )
    path = tmp_path / trajectory_filename("mini_swe_agent", "test/model", problem)
    path.write_text(
        json.dumps({"harness": "mini_swe_agent", "instance_id": "task-1", "problem": problem, "messages": [{}]}),
        encoding="utf-8",
    )
    generated = []

    class FakeAgent:
        def __init__(self, config):
            pass

        def run(self, task):
            generated.append(task)
            return {"output_path": str(path)}

    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setattr(run, "load_instances", lambda path, selector: [row])
    monkeypatch.setitem(run.AGENTS, "mini_swe_agent", FakeAgent)
    monkeypatch.setattr(
        sys, "argv", ["agents.run", "--config", str(config_path), "--output-dir", str(tmp_path)]
    )

    run.main()
    assert generated == []
    assert capsys.readouterr().out.splitlines()[-1] == str(path)

    path.write_text(json.dumps({"instance_id": "another-task"}), encoding="utf-8")
    run.main()
    assert generated == [problem]


def test_batch_instances_run_concurrently_with_isolated_configs(
    tmp_path, monkeypatch, capsys
):
    rows = [
        {
            "instance_id": "task-1",
            "problem_statement": "Fix one.",
            "image": "image-1",
        },
        {
            "instance_id": "task-2",
            "problem_statement": "Fix two.",
            "image": "image-2",
        },
    ]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "generation:\n"
        "  harness: mini_swe_agent\n"
        "  model: test/model\n"
        "  runtime: docker\n"
        "  benchmark_path: unused\n"
        "  instance: -1\n"
        "  n_workers: 2\n",
        encoding="utf-8",
    )
    barrier = threading.Barrier(2)
    worker_configs = []

    class FakeAgent:
        def __init__(self, config):
            self.config = config

        def run(self, task):
            worker_configs.append(
                (self.config.instance_id, self.config.docker_image, task)
            )
            barrier.wait(timeout=3)
            path = tmp_path / f"{self.config.instance_id}.json"
            path.write_text("{}", encoding="utf-8")
            return {"output_path": str(path)}

    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setattr(run, "load_instances", lambda path, selector: rows)
    monkeypatch.setitem(run.AGENTS, "mini_swe_agent", FakeAgent)
    monkeypatch.setattr(
        sys,
        "argv",
        ["agents.run", "--config", str(config_path), "--output-dir", str(tmp_path)],
    )

    run.main()

    assert sorted(worker_configs) == [
        ("task-1", "image-1", "Fix one."),
        ("task-2", "image-2", "Fix two."),
    ]
    assert "2/2" in capsys.readouterr().err


def test_batch_saves_successes_before_reporting_failures(tmp_path, monkeypatch, capsys):
    rows = [
        {"instance_id": "good", "problem_statement": "Fix good."},
        {"instance_id": "bad", "problem_statement": "Fix bad."},
    ]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "generation:\n"
        "  harness: mini_swe_agent\n"
        "  model: test/model\n"
        "  benchmark_path: unused\n"
        "  instance: -1\n"
        "  n_workers: 2\n",
        encoding="utf-8",
    )
    completed = []

    class FakeAgent:
        def __init__(self, config):
            self.config = config

        def run(self, task):
            completed.append(self.config.instance_id)
            if self.config.instance_id == "bad":
                raise subprocess.CalledProcessError(
                    125, ["docker", "run"], stderr="docker failed clearly"
                )
            path = tmp_path / trajectory_filename(
                "mini_swe_agent", "test/model", task
            )
            path.write_text(
                json.dumps(
                    {
                        "harness": "mini_swe_agent",
                        "instance_id": "good",
                        "problem": task,
                        "messages": [],
                    }
                ),
                encoding="utf-8",
            )
            return {"output_path": str(path)}

    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setattr(run, "load_instances", lambda path, selector: rows)
    monkeypatch.setitem(run.AGENTS, "mini_swe_agent", FakeAgent)
    monkeypatch.setattr(
        sys,
        "argv",
        ["agents.run", "--config", str(config_path), "--output-dir", str(tmp_path)],
    )

    for _ in range(2):
        try:
            run.main()
        except SystemExit as exc:
            assert "rerun the same command to resume" in str(exc)
        else:
            raise AssertionError("partial batch failure should be reported")

    assert completed.count("good") == 1
    assert completed.count("bad") == 2
    captured = capsys.readouterr()
    assert "reusing trajectory:" in captured.out
    assert "docker failed clearly" in captured.err


def test_agent_save_atomically_replaces_trajectory(tmp_path, monkeypatch):
    config = AgentConfig(
        harness="mini_swe_agent",
        model="test/model",
        output_dir=tmp_path,
    )
    agent = object.__new__(MiniSWEAgent)
    agent.config = config
    replacements = []
    real_replace = os.replace

    def record_replace(source, destination):
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(
        "agentic_review_annotation_distilabel.agents.base.os.replace",
        record_replace,
    )

    path = agent.save("Fix it.", {"messages": []})

    assert len(replacements) == 1
    assert Path(replacements[0][0]).parent == tmp_path
    assert replacements[0][1] == path
    assert json.loads(path.read_text(encoding="utf-8")) == {"messages": []}
    assert list(tmp_path.glob("*.tmp")) == []


def test_container_platform_comes_from_container(monkeypatch):
    class FakeEnvironment:
        def execute(self, action):
            assert action == {"command": "uname -s; uname -r; uname -v; uname -m"}
            return {
                "returncode": 0,
                "output": "Linux\n6.8.0\n#1 SMP\nx86_64\n",
            }

    assert _container_platform(FakeEnvironment()) == {
        "system": "Linux",
        "release": "6.8.0",
        "version": "#1 SMP",
        "machine": "x86_64",
    }
    assert _docker_run_args("mac") == ["--rm", "--platform", "linux/amd64"]
    assert _docker_run_args("linux") == ["--rm"]
    monkeypatch.setattr(
        "agentic_review_annotation_distilabel.agents.mini_swe_agent.sys.platform",
        "darwin",
    )
    assert _docker_run_args("auto") == ["--rm", "--platform", "linux/amd64"]


def test_loads_first_swebench_image(tmp_path, monkeypatch):
    parquet = tmp_path / "part.parquet"
    parquet.write_text("fixture", encoding="utf-8")

    row = {
        "instance_id": "astropy__astropy-12907",
        "problem_statement": "fix it",
    }

    class FakeRow:
        def __init__(self, data):
            self._data = data

        def to_dict(self):
            return self._data

    class FakeRows:
        empty = False

        def __init__(self, rows):
            self._rows = rows
            self.iloc = self

        def __getitem__(self, item):
            if isinstance(item, int):
                return FakeRow(self._rows[item])
            if item == "instance_id":
                return [candidate["instance_id"] for candidate in self._rows]
            return FakeRows(
                [
                    candidate
                    for candidate, include in zip(self._rows, item, strict=True)
                    if include
                ]
            )

        def to_dict(self):
            return self._rows[0]

    fake_pandas = types.SimpleNamespace(
        read_parquet=lambda path: FakeRows([row]),
        concat=lambda frames: next(iter(frames)),
    )
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)

    row = load_instance(tmp_path, 0)

    assert row["instance_id"] == "astropy__astropy-12907"
    assert swebench_image(row) == (
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    )


def test_opencollab_runner_exports_sdk_artifacts_as_portable_json(
    tmp_path, monkeypatch
):
    fake_package = types.ModuleType("opencollab")
    fake_package.__path__ = []
    fake_package.__version__ = "0.7.0"
    fake_package.OpenCollab = object
    fake_adapters = types.ModuleType("opencollab.adapters")
    fake_adapters.__path__ = []
    fake_env = types.ModuleType("opencollab.adapters.env")
    fake_env.DockerEnvironment = object
    monkeypatch.setitem(sys.modules, "opencollab", fake_package)
    monkeypatch.setitem(sys.modules, "opencollab.adapters", fake_adapters)
    monkeypatch.setitem(sys.modules, "opencollab.adapters.env", fake_env)

    class FakeClient:
        async def team(self, problem, **kwargs):
            assert problem == "Fix it."
            artifacts = kwargs["artifacts"]
            artifacts.mkdir()
            records = [
                {
                    "step": 1,
                    "type": "llm_call",
                    "payload": {"aid": 0, "role": "lead", "content": "Done."},
                }
            ]
            (artifacts / "trajectory.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            (artifacts / "team.json").write_text(
                json.dumps({"sessions": [{"aid": 0, "role": "lead"}]}),
                encoding="utf-8",
            )
            return types.SimpleNamespace(
                status="completed",
                reason=None,
                output="Done.",
                tokens=12,
                metrics={"steps": 1, "sessions": 1},
                agent_failures=(),
            )

    config = AgentConfig(
        harness="opencollab",
        model="test-model",
        runtime="local",
        workspace=tmp_path,
        instance_id="task-1",
        output_dir=tmp_path / "output",
        harness_kwargs={"mode": "team", "provider": "openai"},
    )
    runner = OpenCollabAgent(config)
    monkeypatch.setattr(runner, "_build_client", lambda environment: FakeClient())

    result = runner.run("Fix it.")

    assert result["harness"] == "opencollab"
    assert result["status"] == "completed"
    assert result["trajectory"][0]["type"] == "llm_call"
    assert result["artifacts"]["manifest"]["sessions"][0]["role"] == "lead"
    assert Path(result["output_path"]).is_file()
