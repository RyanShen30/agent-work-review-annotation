from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .base import Agent, AgentConfig

THIRDPARTY = Path(__file__).resolve().parents[1] / "thirdparty" / "openhands"


class OpenHandsAgent(Agent):
    harness_name = "openhands"

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        try:
            from openhands.sdk import LLM
            from openhands.tools.preset.default import get_default_agent
        except ImportError as exc:
            raise RuntimeError(
                f"OpenHands is not installed; initialize submodules and run `uv sync` in {THIRDPARTY}`"
            ) from exc

        self._agent = get_default_agent(
            LLM(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                num_retries=0,
                **config.model_kwargs,
            ),
            cli_mode=True,
        )

    def run(self, problem: str) -> dict[str, Any]:
        from openhands.sdk import Conversation

        workspace: Any = str(self.config.workspace.resolve())
        environment_kwargs = dict(self.config.environment_kwargs)
        keep_image = environment_kwargs.pop("keep_image", False)
        environment_kwargs.pop("save_final_snapshot", None)
        if self.config.runtime == "docker":
            from openhands.workspace import DockerDevWorkspace

            # OpenHands omits --platform for local buildx --load builds.
            os.environ.setdefault("DOCKER_DEFAULT_PLATFORM", "linux/amd64")
            # The coding preset has no browser tool, so skip its optional Chromium preload.
            os.environ["OH_PRELOAD_TOOLS"] = "false"
            forward_env = [
                *environment_kwargs.pop("forward_env", ["DEBUG"]),
                "OH_PRELOAD_TOOLS",
            ]
            workspace = DockerDevWorkspace(
                base_image=self.config.docker_image,
                target="source-minimal",
                platform="linux/amd64",
                working_dir=environment_kwargs.pop("cwd", "/testbed"),
                forward_env=forward_env,
                cleanup_image=not keep_image,
                **environment_kwargs,
            )

        conversation = Conversation(
            agent=self._agent,
            workspace=workspace,
            max_iteration_per_run=self.config.step_limit or 500,
            persistence_dir=None,
            visualizer=None,
        )
        try:
            conversation.send_message(problem)
            conversation.run()
            result = {
                "harness": self.harness_name,
                "instance_id": self.config.instance_id,
                "problem": problem,
                "patch": self._patch(conversation.state.workspace),
                "status": str(conversation.state.execution_status),
                "events": [
                    event.model_dump(mode="json") for event in conversation.state.events
                ],
                "metrics": conversation.conversation_stats.get_combined_metrics().model_dump(
                    mode="json"
                ),
            }
            result["output_path"] = str(self.save(problem, result))
            return result
        finally:
            conversation.close()
            cleanup = getattr(workspace, "cleanup", None)
            if cleanup:
                cleanup()
            if self.config.runtime == "docker" and not keep_image:
                logging.getLogger(__name__).info(
                    "Removing image %s", self.config.docker_image
                )
                result = subprocess.run(
                    ["docker", "image", "rm", "-f", self.config.docker_image],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode:
                    logging.getLogger(__name__).warning(
                        "Could not remove image %s: %s",
                        self.config.docker_image,
                        result.stderr.strip(),
                    )

    def _patch(self, workspace: Any) -> str | None:
        if not self.config.base_commit:
            return None
        cwd = shlex.quote(self.config.environment_kwargs.get("cwd", "/testbed"))
        try:
            result = workspace.execute_command(
                f"git -c safe.directory={cwd} --no-pager diff --no-color "
                f"{self.config.base_commit}",
                timeout=self.config.command_timeout,
            )
        except Exception:
            logging.getLogger(__name__).exception("Could not collect final patch")
            return None
        if result.exit_code:
            logging.getLogger(__name__).warning(
                "Could not collect final patch: %s", result.stderr
            )
            return None
        return result.stdout
