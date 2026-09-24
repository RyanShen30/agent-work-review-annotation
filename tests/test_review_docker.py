import json
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from agentic_review_annotation_distilabel.annotation.annotators import ANNOTATION_AGENTS
from agentic_review_annotation_distilabel.pipelines import (
    DistilabelPipelineConfig,
    run_annotation_pipeline,
)


@pytest.mark.parametrize("image_head", ["abc1234", "def5678"])
def test_docker_review_rebuilds_patch_uses_tools_and_cleans_up(
    monkeypatch, tmp_path, image_head
):
    calls = []
    api_barrier = threading.Barrier(len(ANNOTATION_AGENTS))

    def fake_run(args, **kwargs):
        calls.append((args, kwargs.get("input")))
        if args[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args, 0, "container-id\n", "")
        if any("rev-parse HEAD" in part for part in args):
            return subprocess.CompletedProcess(args, 0, f"{image_head}\n", "")
        if "pwd" in args:
            return subprocess.CompletedProcess(args, 0, "/testbed\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    class ToolCall:
        id = "call-1"
        function = SimpleNamespace(name="run_command", arguments='{"command":"pwd"}')

        def model_dump(self, **_):
            return {
                "id": self.id,
                "type": "function",
                "function": {
                    "name": self.function.name,
                    "arguments": self.function.arguments,
                },
            }

    class FakeCompletions:
        def create(self, **request):
            if not any(message["role"] == "tool" for message in request["messages"]):
                api_barrier.wait(timeout=3)
                answer = SimpleNamespace(content=None, tool_calls=[ToolCall()])
            else:
                assert "exit_code: 0" in request["messages"][-1]["content"]
                is_efficiency = "efficiency-only" in request["messages"][0]["content"]
                answer = SimpleNamespace(
                    content=json.dumps(
                        {
                            "instance_id": "sample",
                            "review_complete": True,
                            "run_review": {
                                "rating": "high" if is_efficiency else "pass",
                            },
                            "step_reviews": [
                                {
                                    "step_id": 1,
                                    "rating": "high" if is_efficiency else "pass",
                                }
                            ],
                        }
                    ),
                    tool_calls=None,
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=answer)])

    class FakeOpenAI:
        def __init__(self, **_):
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    row = {
        "instance_id": "sample",
        "review_workspace": {
            "image": "swebench/sweb.eval.x86_64.example:latest",
            "cwd": "/testbed",
            "base_commit": "abc1234",
        },
        "generated_patch": "diff --git a/a.py b/a.py\n",
        "canonical_steps": [{"step_id": 1}],
        "valid_step_ids": [1],
        "annotator_instructions": {
            agent.name: agent.name for agent in ANNOTATION_AGENTS
        },
    }
    config = DistilabelPipelineConfig(
        runner="llm",
        runtime="docker",
        model="test-model",
        api_key="test-key",
        base_url=None,
        temperature=0.0,
        max_new_tokens=128,
        timeout_seconds=1,
        max_retries=0,
        n_workers=4,
        extra_body=None,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        use_cache=False,
    )

    result = run_annotation_pipeline([row], config)[0]

    assert json.loads(result["generation"])["task_completion_quality"] == {
        "findings": [],
        "run_review": {"rating": "pass"},
    }
    assert sum(args[:2] == ["docker", "run"] for args, _ in calls) == 4
    assert all(
        args[args.index("--platform") + 1] == "linux/amd64"
        for args, _ in calls
        if args[:2] == ["docker", "run"]
    )
    assert sum(args[:2] == ["docker", "rm"] for args, _ in calls) == 4
    assert sum(input_text == row["generated_patch"] for _, input_text in calls) == 4
    assert sum("pwd" in args for args, _ in calls) == 4
    assert sum(
        any("reset --hard abc1234" in part for part in args) for args, _ in calls
    ) == (4 if image_head != "abc1234" else 0)
