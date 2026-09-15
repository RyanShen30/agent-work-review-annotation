import json
import sys
import types
from pathlib import Path

from agentic_review_annotation_distilabel.agents import run
from agentic_review_annotation_distilabel.agents.base import AgentConfig
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
