from __future__ import annotations

import asyncio
import json
import logging
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base import PROJECT_ROOT, Agent, AgentConfig

THIRDPARTY = Path(__file__).resolve().parents[1] / "thirdparty" / "opencollab"
_OPENCOLLAB_CONFIG_KEYS = {
    "wire_protocol",
    "budget",
    "temperature",
    "top_p",
    "max_output_tokens",
    "context_window",
    "thinking",
    "thinking_params",
    "reasoning_effort",
    "llm_max_retries",
    "llm_timeout",
    "llm_connect_timeout",
    "llm_first_event_timeout",
    "llm_stream_idle_timeout",
    "provider_error_time_budget",
    "filter_messages",
}


class OpenCollabAgent(Agent):
    """Run OpenCollab through its public SDK and export one portable JSON record."""

    harness_name = "opencollab"

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        try:
            import opencollab
            from opencollab import OpenCollab
            from opencollab.adapters.env import DockerEnvironment
        except ImportError as exc:
            raise RuntimeError(
                "OpenCollab is not installed; initialize submodules and run "
                f"`uv sync` in {THIRDPARTY}"
            ) from exc

        self._OpenCollab = OpenCollab
        self._DockerEnvironment = DockerEnvironment
        self._version = getattr(opencollab, "__version__", None)
        self._harness_kwargs = dict(config.harness_kwargs)
        self._mode = str(self._harness_kwargs.pop("mode", "team"))
        if self._mode not in {"agent", "team"}:
            raise ValueError(
                "OpenCollab harness_kwargs.mode must be 'agent' or 'team'."
            )

    def run(self, problem: str) -> dict[str, Any]:
        return asyncio.run(self._run(problem))

    async def _run(self, problem: str) -> dict[str, Any]:
        output_dir = self.config.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="opencollab-artifacts-") as temporary:
            artifacts = Path(temporary) / "run"
            environment = self._build_environment()
            client = self._build_client(environment)
            try:
                sdk_result = await self._run_sdk(client, problem, artifacts)
                patch = await self._patch(environment)
                trajectory = _read_jsonl(artifacts / "trajectory.jsonl")
                manifest = _read_manifest(artifacts, self._mode)
            finally:
                if environment is not None:
                    try:
                        await environment.cleanup()
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "Could not clean up the OpenCollab environment"
                        )

        result = {
            "harness": self.harness_name,
            "opencollab_version": self._version,
            "mode": self._mode,
            "instance_id": self.config.instance_id,
            "problem": problem,
            "model": self.config.model,
            "provider": self._provider(),
            "patch": patch,
            "generated_patch": patch,
            "status": sdk_result.status,
            "reason": sdk_result.reason,
            "output": sdk_result.output,
            "tokens": sdk_result.tokens,
            "metrics": sdk_result.metrics,
            "agent_failures": list(sdk_result.agent_failures),
            "trajectory_format": "opencollab.trajectory.jsonl",
            "trajectory": trajectory,
            "artifacts": {"manifest": manifest},
            "swebench": self.config.benchmark_instance,
        }
        result["output_path"] = str(self.save(problem, result))
        return result

    def _build_client(self, environment: Any | None) -> Any:
        raw_model_config = dict(self.config.model_kwargs)
        raw_model_config.pop("drop_params", None)
        client_config = {
            key: value
            for key, value in raw_model_config.items()
            if key in _OPENCOLLAB_CONFIG_KEYS
        }
        unknown = sorted(set(raw_model_config) - _OPENCOLLAB_CONFIG_KEYS)
        if unknown:
            logging.getLogger(__name__).warning(
                "Ignoring unsupported OpenCollab model_kwargs: %s", ", ".join(unknown)
            )
        return self._OpenCollab(
            workspace=self.config.workspace.resolve(),
            model=self.config.model,
            provider=self._provider(),
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            config=client_config,
            environment=environment,
        )

    def _build_environment(self) -> Any | None:
        if self.config.runtime == "local":
            return None
        environment_kwargs = dict(self.config.environment_kwargs)
        workspace = str(
            environment_kwargs.pop(
                "cwd", environment_kwargs.pop("workspace", "/testbed")
            )
        )
        command_prefix = environment_kwargs.pop("command_prefix", None)
        timeout_returncode = int(environment_kwargs.pop("timeout_returncode", -1))
        for ignored in ("keep_image", "pull_timeout", "run_args"):
            environment_kwargs.pop(ignored, None)
        if environment_kwargs:
            logging.getLogger(__name__).warning(
                "Ignoring unsupported OpenCollab environment_kwargs: %s",
                ", ".join(sorted(environment_kwargs)),
            )
        return self._DockerEnvironment(
            image=self.config.docker_image,
            workspace=workspace,
            exec_workdir=workspace,
            command_prefix=command_prefix,
            timeout_returncode=timeout_returncode,
        )

    async def _run_sdk(self, client: Any, problem: str, artifacts: Path) -> Any:
        kwargs = dict(self._harness_kwargs)
        provider = kwargs.pop("provider", None)
        if provider is not None and str(provider) != self._provider():
            raise ValueError("OpenCollab provider must be configured in one place.")
        budget = int(kwargs.pop("budget", 1_000_000))
        timeout = kwargs.pop("timeout", None)
        cleanup_timeout = float(kwargs.pop("cleanup_timeout", 5.0))

        if self._mode == "agent":
            for team_only in (
                "team_config",
                "use_worktrees",
                "prebuild_team",
                "allow_unisolated_shell",
                "serialize_turns",
            ):
                kwargs.pop(team_only, None)
            allowed = {"tools", "name", "system_prompt"}
            _reject_unknown(kwargs, allowed, mode=self._mode)
            return await client.agent(
                problem,
                tools=kwargs.pop("tools", "coding"),
                name=str(kwargs.pop("name", "agent")),
                system_prompt=kwargs.pop("system_prompt", None),
                budget=budget,
                max_steps=self.config.step_limit or 100,
                timeout=timeout,
                cleanup_timeout=cleanup_timeout,
                artifacts=artifacts,
                trace=True,
            )

        allowed = {
            "team_config",
            "use_worktrees",
            "prebuild_team",
            "allow_unisolated_shell",
            "serialize_turns",
        }
        _reject_unknown(kwargs, allowed, mode=self._mode)
        team_config = kwargs.pop("team_config", None)
        if team_config is not None:
            team_config = Path(team_config)
            if not team_config.is_absolute():
                team_config = PROJECT_ROOT / team_config
        return await client.team(
            problem,
            config=team_config,
            budget=budget,
            max_steps=self.config.step_limit or 100,
            timeout=timeout,
            cleanup_timeout=cleanup_timeout,
            artifacts=artifacts,
            trace=True,
            use_worktrees=bool(kwargs.pop("use_worktrees", True)),
            prebuild_team=bool(kwargs.pop("prebuild_team", False)),
            allow_unisolated_shell=kwargs.pop("allow_unisolated_shell", True),
            serialize_turns=bool(kwargs.pop("serialize_turns", False)),
        )

    def _provider(self) -> str:
        return str(self.config.harness_kwargs.get("provider", "openai"))

    async def _patch(self, environment: Any | None) -> str | None:
        if not self.config.base_commit:
            return None
        command = (
            "git -c safe.directory=* --no-pager diff --no-color "
            f"{shlex.quote(self.config.base_commit)}"
        )
        if environment is not None:
            result = await environment.exec_cmd(
                command, timeout=self.config.command_timeout
            )
            if result.returncode:
                logging.getLogger(__name__).warning(
                    "Could not collect OpenCollab patch: %s", result.stderr
                )
                return None
            return result.stdout
        completed = await asyncio.to_thread(
            subprocess.run,
            [
                "git",
                "-c",
                "safe.directory=*",
                "--no-pager",
                "diff",
                "--no-color",
                self.config.base_commit,
            ],
            cwd=self.config.workspace.resolve(),
            capture_output=True,
            text=True,
            timeout=self.config.command_timeout,
            check=False,
        )
        if completed.returncode:
            logging.getLogger(__name__).warning(
                "Could not collect OpenCollab patch: %s", completed.stderr.strip()
            )
            return None
        return completed.stdout


def _reject_unknown(kwargs: dict[str, Any], allowed: set[str], *, mode: str) -> None:
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported OpenCollab {mode} harness_kwargs: {', '.join(unknown)}"
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(
            f"OpenCollab did not create required trajectory: {path.name}"
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(
                f"OpenCollab trajectory line {line_number} is not an object."
            )
        records.append(value)
    if not records:
        raise RuntimeError("OpenCollab created an empty trajectory.")
    return records


def _read_manifest(artifacts: Path, mode: str) -> dict[str, Any] | None:
    path = artifacts / ("team.json" if mode == "team" else "agent.json")
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None
