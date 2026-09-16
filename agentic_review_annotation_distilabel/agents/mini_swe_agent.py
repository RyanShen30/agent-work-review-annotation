from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .base import Agent, AgentConfig

THIRDPARTY = Path(__file__).resolve().parents[1] / "thirdparty" / "mini-swe-agent"
DEFAULT_CONFIG = THIRDPARTY / "src" / "minisweagent" / "config" / "mini.yaml"


class MiniSWEAgent(Agent):
    harness_name = "mini_swe_agent"

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        try:
            from minisweagent.agents import get_agent
            from minisweagent.environments import get_environment
            from minisweagent.models import get_model
        except ImportError as exc:
            raise RuntimeError(
                "Please clone the mini-swe-agent submodule first"
            ) from exc

        raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
        logging.getLogger("minisweagent").setLevel(logging.INFO)
        os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
        model_config = raw["model"] | {"model_name": config.model}
        model_config["model_kwargs"] = (
            raw["model"].get("model_kwargs", {}) | config.model_kwargs
        )
        if config.api_key:
            model_config["model_kwargs"]["api_key"] = config.api_key
        if config.base_url:
            model_config["model_kwargs"]["api_base"] = config.base_url

        env_config = raw.get("environment", {}) | config.environment_kwargs
        self._keep_image = env_config.pop("keep_image", False)
        self._save_final_snapshot = env_config.pop(
            "save_final_snapshot", config.runtime == "docker"
        )
        env_config |= {
            "environment_class": config.runtime,
            "timeout": config.command_timeout,
        }
        env_config.setdefault(
            "cwd",
            "/workspace"
            if config.runtime == "docker"
            else str(config.workspace.resolve()),
        )
        if config.runtime == "docker":
            env_config["image"] = config.docker_image
            env_config.setdefault("run_args", ["--rm", "--platform", "linux/amd64"])

        agent_config = raw["agent"] | {
            "agent_class": "default",
            "mode": "yolo",
            "step_limit": config.step_limit,
            "cost_limit": config.cost_limit,
            "output_path": None,
        }
        logging.getLogger("minisweagent").info(
            "Starting %s environment%s",
            config.runtime,
            f" with image {config.docker_image}" if config.docker_image else "",
        )
        self._env = get_environment(env_config)
        self._agent = get_agent(get_model(config=model_config), self._env, agent_config)

    def run(self, problem: str) -> dict[str, Any]:
        try:
            outcome = self._agent.run(problem)
            result = self._agent.serialize(
                {
                    "harness": self.harness_name,
                    "instance_id": self.config.instance_id,
                    "problem": problem,
                    "patch": self._patch(),
                    "outcome": outcome,
                    "swebench": self.config.benchmark_instance,
                }
            )
            review_workspace = self._capture_final_snapshot(problem)
            if review_workspace:
                result["review_workspace"] = review_workspace
            result["output_path"] = str(self.save(problem, result))
            return result
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        cleanup = getattr(self._env, "cleanup", None)
        if self.config.runtime != "docker":
            if cleanup:
                cleanup()
            return

        docker = self._env.config.executable
        container_id = getattr(self._env, "container_id", None)
        if container_id:
            subprocess.run(
                [docker, "rm", "-f", container_id],
                capture_output=True,
                check=False,
            )
            self._env.container_id = None

        if self._keep_image:
            logging.getLogger("minisweagent").info(
                "Keeping image %s", self.config.docker_image
            )
            return

        logging.getLogger("minisweagent").info(
            "Removing image %s", self.config.docker_image
        )
        result = subprocess.run(
            [docker, "image", "rm", "-f", self.config.docker_image],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            logging.getLogger("minisweagent").warning(
                "Could not remove image %s: %s",
                self.config.docker_image,
                result.stderr.strip(),
            )

    def _patch(self) -> str | None:
        if not self.config.base_commit:
            return None
        result = self._env.execute(
            {"command": f"git --no-pager diff --no-color {self.config.base_commit}"}
        )
        return result.get("output") if result.get("returncode") == 0 else None

    def _capture_final_snapshot(self, problem: str) -> dict[str, Any] | None:
        if self.config.runtime != "docker" or not self._save_final_snapshot:
            return None

        docker = self._env.config.executable
        container_id = getattr(self._env, "container_id", None)
        if not container_id:
            raise RuntimeError("Cannot save final snapshot without a running container")

        instance = self.config.instance_id or problem
        slug = re.sub(r"[^a-z0-9_.-]+", "-", instance.lower()).strip("-._")
        slug = slug[:48] or "run"
        snapshot_image = f"agent-work-review/final:{slug}-{container_id[:12]}"
        logging.getLogger("minisweagent").info(
            "Saving final coding workspace as %s", snapshot_image
        )
        committed = subprocess.run(
            [docker, "commit", "--pause=true", container_id, snapshot_image],
            capture_output=True,
            text=True,
            timeout=max(self.config.command_timeout, 900),
            check=False,
        )
        if committed.returncode:
            raise RuntimeError(
                f"Could not save final coding workspace: {committed.stderr.strip()}"
            )

        snapshot_id = committed.stdout.strip()
        if not snapshot_id:
            raise RuntimeError("Docker commit returned no final snapshot ID")
        cleanup_images = [snapshot_image]
        if self._keep_image and self.config.docker_image:
            cleanup_images.append(self.config.docker_image)
        return {
            "image": self.config.docker_image,
            "snapshot_image": snapshot_image,
            "snapshot_id": snapshot_id,
            "snapshot_kind": "agent_final",
            "cleanup_images": cleanup_images,
        }
