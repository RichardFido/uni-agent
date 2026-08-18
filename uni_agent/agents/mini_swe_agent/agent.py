"""mini-swe-agent: a black-box agent that runs the real mini-swe-agent inside the sandbox.

mini-swe-agent is launched *in* the sandbox from a prebuilt tool image (venv at
``/opt/mini-swe-agent``) whose ``bin/run_agent.py`` reads a task config from
**stdin** and writes the result JSON to **stdout**. This agent builds that
config (with the gateway URL rewritten to the sandbox-internal tunnel), pipes
it in via base64, and parses the result JSON out of stdout.

Reference: https://github.com/SWE-agent/mini-swe-agent
"""

from __future__ import annotations

import base64
import json
import logging
import shlex
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ...sandbox.tunnel import DEFAULT_PROXY_PORT, rewrite_gateway_url
from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

TOOL_PYTHON = "/opt/mini-swe-agent/bin/python"
RUN_AGENT_SCRIPT = "/opt/mini-swe-agent/bin/run_agent.py"


def build_agent_command(
    *,
    config_b64: str,
    conda_env: str = "testbed",
    tool_python: str = TOOL_PYTHON,
    run_agent_script: str = RUN_AGENT_SCRIPT,
) -> str:
    """Build the shell command that runs ``run_agent.py`` inside the sandbox.

    The task config is piped via base64-encoded stdin (the protocol ``run_agent.py``
    expects). The tool python is called through the task's conda env so mini-swe-agent
    resolves the repo environment inside ``/testbed``.
    """
    conda_prefix = f"/opt/miniconda3/envs/{conda_env}"
    run_agent_env = (
        f"CONDA_DEFAULT_ENV={shlex.quote(conda_env)} "
        f"CONDA_PREFIX={shlex.quote(conda_prefix)} "
        f"PATH={shlex.quote(conda_prefix + '/bin')}:/opt/miniconda3/bin:$PATH "
        "PIP_DISABLE_PIP_VERSION_CHECK=1 "
        "PIP_PROGRESS_BAR=off"
    )
    return (
        "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy; "
        f"printf %s {shlex.quote(config_b64)} | base64 -d | "
        f"env {run_agent_env} {tool_python} {run_agent_script}"
    )


def parse_agent_result(stdout: str) -> dict[str, Any]:
    """Parse the result JSON from ``run_agent.py``'s stdout.

    litellm may print error/noise lines to stdout, polluting the output, so the last
    line starting with ``{`` wins; fall back to the whole stdout, then to an error marker.
    """
    stdout = stdout.strip()
    if not stdout:
        return {"exit_status": "error", "submission": ""}
    for line in reversed([ln.strip() for ln in stdout.split("\n") if ln.strip()]):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning("mini_swe_agent: failed to parse agent result (stdout tail): %.1000s", stdout)
        return {"exit_status": "error", "submission": ""}


class MiniSweAgentConfig(AgentConfig):
    """Black-box launch params for mini-swe-agent (endpoint lives on :attr:`AgentConfig.model`)."""

    name: str = "mini_swe_agent"
    step_limit: int = Field(default=100, description="mini-swe-agent max agent steps.")
    run_timeout: float = Field(default=7200.0, description="Wallclock cap (s) on the agent process.")
    conda_env: str = Field(default="testbed", description="Task repo conda env, activated around the launch.")
    proxy_port: int = Field(
        default=DEFAULT_PROXY_PORT,
        description="Sandbox-internal tunnel port (must match the sandbox's proxy_port).",
    )
    tool_python: str = Field(
        default=TOOL_PYTHON,
        description="Prebuilt tool-image python that runs run_agent.py.",
    )
    run_agent_script: str = Field(
        default=RUN_AGENT_SCRIPT,
        description="In-tool-image entrypoint script reading config from stdin.",
    )


@register_agent("mini_swe_agent")
class MiniSweAgentAgent(Agent):
    """Black-box solver: launch mini-swe-agent in the sandbox against ``config.model``."""

    config_model = MiniSweAgentConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        cfg: MiniSweAgentConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError("mini_swe_agent: config.model.base_url is not set (the gateway/vLLM policy endpoint)")
        task = self._extract_task(messages)

        # 1) Build the task config; the gateway URL is rewritten to the sandbox-internal tunnel.
        task_config = {
            "task": task,
            "gateway_url": rewrite_gateway_url(cfg.model.base_url, cfg.proxy_port),
            "agent": {"step_limit": cfg.step_limit},
        }
        config_b64 = base64.b64encode(json.dumps(task_config).encode()).decode()

        # 2) Pipe it into the prebuilt tool-image python.
        agent_cmd = build_agent_command(
            config_b64=config_b64,
            conda_env=cfg.conda_env,
            tool_python=cfg.tool_python,
            run_agent_script=cfg.run_agent_script,
        )
        result = await sandbox.exec_shell(agent_cmd, timeout=cfg.run_timeout)

        # 3) Parse the result JSON from stdout (litellm noise tolerated).
        agent_info = parse_agent_result(result.stdout or "")
        logger.info(
            "mini_swe_agent: done exit_status=%s submission=%d chars rc=%s",
            agent_info.get("exit_status"),
            len(agent_info.get("submission", "")),
            result.exit_code,
        )
        return AgentResult(
            output=agent_info,
            transcript=list(messages),
            info={"step_limit": cfg.step_limit, "exit_status": agent_info.get("exit_status")},
            # Explicit completion: mini-swe-agent reports "Submitted" only when it
            # produced a submission; anything else (error/timeout) is "not finished",
            # so those episodes can be masked from the loss via
            # mask_unfinished_episode=True in the framework config.
            finished=agent_info.get("exit_status") == "Submitted",
        )

    @staticmethod
    def _extract_task(messages: list[dict[str, Any]]) -> str:
        if len(messages) > 2:
            raise ValueError(f"mini_swe_agent accepts at most 2 messages (system?, user), got {len(messages)}")
        problem = next((m["content"] for m in messages if m.get("role") == "user"), None)
        if not problem:
            raise ValueError("mini_swe_agent requires a 'user' message (the problem statement)")
        return problem
