import json
import subprocess
from types import SimpleNamespace

import main
from agentic_review_annotation_distilabel.agents.base import AgentConfig
from agentic_review_annotation_distilabel.agents.mini_swe_agent import MiniSWEAgent
from agentic_review_annotation_distilabel.pipelines.review_docker import (
    review_container,
)


def test_mini_swe_agent_commits_final_snapshot_before_container_cleanup(
    monkeypatch, tmp_path
):
    config = AgentConfig(
        harness="mini_swe_agent",
        model="test/model",
        runtime="docker",
        docker_image="swebench/base:latest",
        instance_id="owner__repo-123",
        base_commit="abc1234",
        output_dir=tmp_path,
        environment_kwargs={"cwd": "/testbed"},
    )
    agent = object.__new__(MiniSWEAgent)
    agent.config = config
    agent._keep_image = True
    agent._save_final_snapshot = True
    agent._env = SimpleNamespace(
        config=SimpleNamespace(executable="docker"),
        container_id="abcdef1234567890",
    )
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["docker", "commit"]:
            return subprocess.CompletedProcess(args, 0, "sha256:final-image\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    workspace = agent._capture_final_snapshot("Fix it")
    path = agent.save("Fix it", {"review_workspace": workspace})
    agent._cleanup()

    assert calls[0][:3] == ["docker", "commit", "--pause=true"]
    assert calls[1] == ["docker", "rm", "-f", "abcdef1234567890"]
    assert all(call[:3] != ["docker", "image", "rm"] for call in calls)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["review_workspace"] == {
        "image": "swebench/base:latest",
        "snapshot_image": ("agent-work-review/final:owner__repo-123-abcdef123456"),
        "snapshot_id": "sha256:final-image",
        "snapshot_kind": "agent_final",
        "cleanup_images": [
            "agent-work-review/final:owner__repo-123-abcdef123456",
            "swebench/base:latest",
        ],
        "cwd": "/testbed",
        "base_commit": "abc1234",
    }


def test_reviewer_prefers_final_snapshot_without_reapplying_patch(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs.get("input")))
        if args[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args, 0, "review-container\n", "")
        if any("rev-parse HEAD" in part for part in args):
            return subprocess.CompletedProcess(args, 0, "abc1234\n", "")
        if "pwd" in args:
            return subprocess.CompletedProcess(args, 0, "/testbed\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    row = {
        "instance_id": "sample",
        "review_workspace": {
            "image": "swebench/base:latest",
            "snapshot_image": "agent-work-review/final:sample",
            "snapshot_id": "sha256:final-image",
            "cwd": "/testbed",
            "base_commit": "abc1234",
        },
        "generated_patch": "diff --git a/a.py b/a.py\n",
    }
    config = SimpleNamespace(
        docker_image=None,
        docker_cwd=None,
        docker_platform=None,
        command_timeout=30,
    )

    with review_container(row, config) as execute:
        assert execute("pwd") == "exit_code: 0\n/testbed\n"

    run_call = next(args for args, _ in calls if args[:2] == ["docker", "run"])
    assert "sha256:final-image" in run_call
    assert not any(input_text == row["generated_patch"] for _, input_text in calls)
    assert not any(
        any(
            "reset --hard" in part or "git -c safe.directory='*' apply" in part
            for part in args
        )
        for args, _ in calls
    )


def test_cleanup_removes_managed_snapshots_before_shared_base(monkeypatch, tmp_path):
    trajectories = []
    for index in (1, 2):
        path = tmp_path / f"run-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "review_workspace": {
                        "image": "swebench/base:latest",
                        "snapshot_image": f"agent-work-review/final:run-{index}",
                        "cleanup_images": [
                            f"agent-work-review/final:run-{index}",
                            "swebench/base:latest",
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )
        trajectories.append(path)
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    main.cleanup_docker_images(trajectories)

    assert calls == [
        ["docker", "image", "rm", "-f", "agent-work-review/final:run-1"],
        ["docker", "image", "rm", "-f", "agent-work-review/final:run-2"],
        ["docker", "image", "rm", "-f", "swebench/base:latest"],
    ]


def test_cleanup_policy_keeps_failed_runs_for_debugging():
    assert main.should_cleanup("always", succeeded=False)
    assert main.should_cleanup("on_success", succeeded=True)
    assert not main.should_cleanup("on_success", succeeded=False)
    assert not main.should_cleanup("never", succeeded=True)
