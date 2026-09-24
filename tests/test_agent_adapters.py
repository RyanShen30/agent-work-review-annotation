import json

import pytest

from agentic_review_annotation_distilabel.adapters import (
    MiniSWEAgentAdapter,
    OpenCollabAdapter,
    OpenHandsAdapter,
)
from agentic_review_annotation_distilabel.run import collect_input_paths
from agentic_review_annotation_distilabel.steps import (
    AgentStepParser,
    MiniSWEAgentStepParser,
    OpenCollabStepParser,
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
    assert steps[0].step_id == 1


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
    assert steps[0].step_id == 1
    assert steps[0].raw_message_indices == [2, 3]
    assert steps[0].action_ids == ["bash"]
    assert steps[0].observation_indices == [3]
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


def test_mini_swe_agent_preserves_swebench_oracle_separately():
    raw = {
        "instance_id": "astropy__astropy-14365",
        "repo": "astropy/astropy",
        "base_commit": "abc123",
        "problem_statement": "Fix QDP parsing.",
        "hints_text": "Commands are case-insensitive.",
        "created_at": "2023-01-01T00:00:00Z",
        "version": "1.0",
        "difficulty": "medium",
        "patch": "gold patch",
        "test_patch": "test patch",
        "FAIL_TO_PASS": '["test_qdp_lowercase"]',
        "PASS_TO_PASS": ["test_existing"],
        "eval_type": "pytest",
        "image": "swebench/image:latest",
        "log_parser": "parse_pytest",
        "eval_script": "pytest",
        "generated_patch": "agent patch",
        "messages": [
            {"role": "user", "content": "Fix QDP parsing."},
            {"role": "assistant", "content": "Done."},
        ],
    }

    sample = MiniSWEAgentAdapter().adapt(raw)

    assert sample.patch == "agent patch"
    assert sample.source == {
        "benchmark": "SWE-bench_Verified",
        "repo": "astropy/astropy",
        "base_commit": "abc123",
        "problem_statement": "Fix QDP parsing.",
        "hints_text": "Commands are case-insensitive.",
        "created_at": "2023-01-01T00:00:00Z",
        "version": "1.0",
        "difficulty": "medium",
    }
    assert sample.oracle == {
        "gold_patch": "gold patch",
        "test_patch": "test patch",
        "fail_to_pass": ["test_qdp_lowercase"],
        "pass_to_pass": ["test_existing"],
        "eval_type": "pytest",
        "eval_image": "swebench/image:latest",
        "eval_script": "pytest",
        "log_parser": "parse_pytest",
    }


def test_opencollab_adapter_preserves_trace_and_swebench_metadata():
    raw = {
        "harness": "opencollab",
        "opencollab_version": "0.7.0",
        "mode": "team",
        "instance_id": "django__django-12345",
        "problem": "Fix the regression.",
        "model": "test-model",
        "provider": "openai",
        "generated_patch": "diff --git a/a.py b/a.py",
        "status": "completed",
        "tokens": 321,
        "metrics": {"steps": 2, "sessions": 2},
        "trajectory": [
            {
                "step": 1,
                "type": "llm_call",
                "payload": {"aid": 0, "role": "lead", "content": "Inspecting."},
            }
        ],
        "swebench": {
            "repo": "django/django",
            "base_commit": "abc123",
            "problem_statement": "Fix the regression.",
            "patch": "gold patch",
            "test_patch": "test patch",
            "FAIL_TO_PASS": '["test_regression"]',
            "PASS_TO_PASS": ["test_existing"],
        },
    }

    sample = OpenCollabAdapter().adapt(raw)

    assert sample.instance_id == "django__django-12345"
    assert sample.task == "Fix the regression."
    assert sample.patch == "diff --git a/a.py b/a.py"
    assert sample.source == {
        "benchmark": "SWE-bench",
        "repo": "django/django",
        "base_commit": "abc123",
        "problem_statement": "Fix the regression.",
    }
    assert sample.run["harness"] == "opencollab"
    assert sample.run["config"]["mode"] == "team"
    assert sample.run["config"]["tokens"] == 321
    assert sample.oracle == {
        "gold_patch": "gold patch",
        "test_patch": "test patch",
        "fail_to_pass": ["test_regression"],
        "pass_to_pass": ["test_existing"],
    }


def test_opencollab_step_parser_tracks_interleaved_agent_turns():
    trace = [
        {
            "step": 1,
            "type": "assigned.topology_nodes",
            "payload": {"entry_role": "lead", "nodes": []},
        },
        {
            "step": 2,
            "type": "llm_call_started",
            "payload": {"aid": 0, "role": "lead", "response_session_id": "lead-1"},
        },
        {
            "step": 3,
            "type": "context_shaping",
            "payload": {"aid": 0, "rung": "none"},
        },
        {
            "step": 4,
            "type": "llm_call",
            "payload": {
                "aid": 0,
                "role": "lead",
                "session_step": 1,
                "response_session_id": "lead-1",
                "content": "I will inspect and delegate.",
                "tool_calls": [
                    {
                        "id": "read-1",
                        "name": "file_read",
                        "arguments": '{"path":"a.py"}',
                    },
                    {"id": "spawn-1", "name": "spawn_agent", "arguments": "{}"},
                ],
            },
        },
        {
            "step": 5,
            "type": "tool_exec",
            "payload": {
                "aid": 0,
                "tool": "file_read",
                "tool_call_id": "read-1",
                "result": "source",
            },
        },
        {
            "step": 6,
            "type": "llm_call",
            "payload": {
                "aid": 1,
                "role": "coder",
                "session_step": 1,
                "content": "I will edit it.",
                "tool_calls": [
                    {"id": "edit-1", "name": "apply_patch", "arguments": "{}"}
                ],
            },
        },
        {
            "step": 7,
            "type": "tool_exec",
            "payload": {
                "aid": 1,
                "tool": "apply_patch",
                "tool_call_id": "edit-1",
                "result": "Done!",
            },
        },
        {
            "step": 8,
            "type": "message_delivered",
            "payload": {"from_aid": 1, "to_aid": 0, "summary": "implemented"},
        },
        {
            "step": 9,
            "type": "tool_exec",
            "payload": {
                "aid": 0,
                "tool": "spawn_agent",
                "tool_call_id": "spawn-1",
                "result": "Agent 1 completed",
            },
        },
    ]
    sample = OpenCollabAdapter().adapt(
        {"task_id": "task-1", "description": "Fix it.", "trajectory": trace}
    )

    steps = OpenCollabStepParser().parse(sample)

    assert len(steps) == 2
    assert steps[0].content["agent"] == {
        "aid": 0,
        "role": "lead",
        "session_step": 1,
    }
    assert steps[0].action_ids == ["read-1", "spawn-1"]
    assert steps[0].observation_indices == [4, 8]
    assert [event["payload"]["tool"] for event in steps[0].content["observations"]] == [
        "file_read",
        "spawn_agent",
    ]
    assert steps[0].content["context_events"] == [trace[0]]
    assert steps[0].content["runtime_events"] == [trace[1], trace[2]]
    assert steps[1].content["agent"]["role"] == "coder"
    assert steps[1].observation_indices == [6]
    assert steps[1].content["orchestration_events"] == [trace[7]]


def test_opencollab_adapter_accepts_embedded_jsonl():
    trajectory = "\n".join(
        json.dumps(record)
        for record in [
            {"type": "llm_call", "payload": {"aid": 0, "content": "Done"}},
            {"type": "session_terminal", "payload": {"aid": 0, "phase": "done"}},
        ]
    )

    sample = OpenCollabAdapter().adapt({"task_id": "jsonl-1", "trajectory": trajectory})

    assert len(sample.trajectory) == 2
    assert OpenCollabStepParser().parse(sample)[0].step_id == 1
