import pytest

from agentic_review_annotation_distilabel.adapters import (
    MiniSWEAgentAdapter,
    OpenHandsAdapter,
)
from agentic_review_annotation_distilabel.steps import AgentStepParser


@pytest.mark.parametrize(
    ("adapter", "raw", "trajectory_key"),
    [
        (
            MiniSWEAgentAdapter(),
            {
                "instance_id": "astropy__astropy-12907",
                "problem": "fix it",
                "messages": [{"role": "user", "content": "fix it"}],
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


def test_requires_instance_id():
    with pytest.raises(ValueError, match="instance_id"):
        MiniSWEAgentAdapter().adapt({"messages": [{"role": "user", "content": "task"}]})
