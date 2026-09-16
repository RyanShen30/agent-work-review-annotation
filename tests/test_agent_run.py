import json
import sys
import types
from pathlib import Path

from agentic_review_annotation_distilabel.agents import run
from agentic_review_annotation_distilabel.agents.base import AgentConfig, trajectory_filename
from agentic_review_annotation_distilabel.agents.opencollab import OpenCollabAgent
from agentic_review_annotation_distilabel.agents.run import (
    load_instance,
    load_instances,
    swebench_image,
)


def test_minus_one_loads_entire_benchmark(monkeypatch):
    rows = [{"instance_id": "one"}, {"instance_id": "two"}]
    monkeypatch.setattr(run, "load_benchmark", lambda path: rows)

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
