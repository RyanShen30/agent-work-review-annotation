import json

import pytest

from agentic_review_annotation_distilabel.adapters import (
    MiniSWEAgentAdapter,
    OpenHandsAdapter,
)
from agentic_review_annotation_distilabel.run import collect_input_paths
from agentic_review_annotation_distilabel.steps import (
    AgentStepParser,
    MiniSWEAgentStepParser,
)


@pytest.mark.parametrize(
    ("adapter", "raw", "trajectory_key"),
    [
        (
            MiniSWEAgentAdapter(),
            {
                "instance_id": "astropy__astropy-12907",
                "problem": "fix it",
                "messages": [
                    {"role": "user", "content": "fix something else"},
                    {"role": "assistant", "content": "I will inspect the repo."},
                    {"role": "tool", "content": "listing"},
                    {"role": "assistant", "content": "Done."},
                ],
                "outcome": {"exit_status": "Submitted"},
                "patch": "diff --git a/a.py b/a.py",
            },
            "messages",
        ),
        (
            OpenHandsAdapter(),
            {
                "instance_id": "astropy__astropy-12907",
                "problem": "fix it",
                "events": [{"kind": "MessageEvent", "source": "user"}],
                "status": "finished",
                "patch": "diff --git a/a.py b/a.py",
            },
            "events",
        ),
    ],
)
def test_adapts_saved_trajectory(adapter, raw, trajectory_key):
    sample = adapter.adapt(raw)
    steps = AgentStepParser().parse(sample)

    assert sample.instance_id == "astropy__astropy-12907"
    assert sample.task == "fix it"
    assert sample.trajectory == raw[trajectory_key]
    assert sample.patch == raw["patch"]
    assert steps[0].step_id == 0


def test_mini_swe_agent_adapter_drops_raw_response_but_keeps_actions():
    sample = MiniSWEAgentAdapter().adapt(
        {
            "instance_id": "repo__repo-1",
            "messages": [
                {
                    "role": "assistant",
                    "provider_specific_fields": {},
                    "extra": {"actions": [{"command": "pytest"}], "response": {"choices": []}},
                }
            ],
        }
    )

    assert sample.trajectory == [
        {"role": "assistant", "extra": {"actions": [{"command": "pytest"}]}}
    ]


def test_mini_swe_agent_adapter_feeds_dedicated_step_parser():
    raw = {
        "info": {
            "instance_id": "django__django-11049",
            "config": {"environment": {"cwd": "/workspace/django"}},
            "model_stats": {"cost": 0.42},
        },
        "problem": "Fix the regression.",
        "messages": [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": "Fix the regression."},
            {
                "role": "assistant",
                "content": "I will inspect the failing code.",
                "tool_calls": [{"function": {"name": "bash", "arguments": "pytest"}}],
            },
            {"role": "tool", "content": "1 failed"},
            {"role": "assistant", "content": "I changed the implementation."},
        ],
        "outcome": {"exit_status": "Submitted"},
        "patch": "diff --git a/django/a.py b/django/a.py",
    }

    sample = MiniSWEAgentAdapter().adapt(raw)
    steps = MiniSWEAgentStepParser().parse(sample)

    assert sample.instance_id == "django__django-11049"
    assert sample.task == "Fix the regression."
    assert sample.evaluation["exit_status"] == "Submitted"
    assert sample.repository == {"instance_id": "django__django-11049", "cwd": "/workspace/django"}
    assert len(steps) == 2
    assert steps[0].content["type"] == "agent_turn"
    assert steps[0].content["actions"] == [{"function": {"name": "bash", "arguments": "pytest"}}]
    assert steps[0].content["observations"] == [{"role": "tool", "content": "1 failed"}]
    assert steps[0].content["context_messages"] == [
        {"role": "system", "content": "system context"},
        {"role": "user", "content": "Fix the regression."},
    ]


def test_requires_instance_id():
    with pytest.raises(ValueError, match="instance_id"):
        MiniSWEAgentAdapter().adapt({"messages": [{"role": "user", "content": "task"}]})


def test_directory_input_filters_other_harnesses(tmp_path):
    mini = tmp_path / "mini.json"
    openhands = tmp_path / "openhands.json"
    mini.write_text(json.dumps({"harness": "mini_swe_agent"}))
    openhands.write_text(json.dumps({"harness": "openhands"}))

    assert collect_input_paths(tmp_path, "mini_swe_agent") == [mini]
