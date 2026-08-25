"""deepseek_harness: a black-box agent that runs the real `dsh --profile headless`
inside the sandbox.

DeepSeek Harness is launched *in* the sandbox from a prebuilt tool image mounted
at ``/opt/dsh`` (Node CLI + node runtime + the baked ``minimal_patch.yml``).
This agent builds the ``dsh --profile headless`` shell command, points DSH's
built-in ``llm-deepseek`` route at the gateway tunnel, execs it in the sandbox,
and returns the result.

Unlike ``mini_swe_agent`` there is **no in-sandbox ``run_agent.py`` entrypoint**:
``dsh`` is itself the agent binary, so this agent builds and runs the command
directly. The gateway reverse tunnel (``upstream`` + ``model.base_url`` rewrite
to ``http://127.0.0.1:<proxy_port>/v1``) and the reward POST are handled by the
unified ``uni_agent.framework.task_runner.run_task`` bridge -- this agent only
owns the launch.

Wire compat (DSH speaks OpenAI Chat Completions streaming + tool_calls through
the gateway OpenAI adapter) is proven by ``examples/deepseek_harness/demo``.
"""

from __future__ import annotations

import json
import logging
import shlex
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

# DSH writes session persistence + the auto-initialized headless profile here;
# must be writable inside the sandbox. Auto-init uses shipped templates and
# needs no network (proven by the demo's isolated DSH_HOME).
DSH_HOME_IN_SANDBOX = "/tmp/dsh-home"


def build_dsh_command(
    *,
    task: str,
    base_url: str,
    conda_env: str = "testbed",
    tool_mode: str = "full",
    dsh_bin: str = "/opt/dsh/bin/dsh",
    minimal_patch_path: str = "/opt/dsh/minimal_patch.yml",
) -> str:
    """Build the shell command that runs DSH headless against the gateway tunnel.

    ``base_url`` is already the sandbox-internal tunnel address
    (``http://127.0.0.1:<proxy_port>/v1``) -- ``run_task`` rewrites
    ``agent.model.base_url`` before this agent runs, so the agent is
    tunnel-agnostic.

    ``tool_mode``:
      * ``"full"``    -- the default 25-tool Code Mode surface.
      * ``"minimal"`` -- apply the baked ``minimal_patch.yml`` to restrict the
        request to ``bash`` + ``str_replace_editor`` (see demo/README for the
        measured 25 -> 2 tool / 4122 -> 257 sys_chars reduction).
    """
    # DSH's llm-deepseek route honors DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY. The
    # gateway has no auth path, so a dummy key is fine.
    env = {
        "DEEPSEEK_BASE_URL": base_url,
        "DEEPSEEK_API_KEY": "not-needed",
        "DSH_HOME": DSH_HOME_IN_SANDBOX,
        "IS_SANDBOX": "1",
        "DISABLE_AUTOUPDATER": "1",
    }
    env_assignments = [f"{key}={shlex.quote(value)}" for key, value in env.items()]

    if conda_env:
        conda_prefix = f"/opt/miniconda3/envs/{conda_env}"
        env_assignments.extend(
            [
                f"CONDA_DEFAULT_ENV={shlex.quote(conda_env)}",
                f"CONDA_PREFIX={shlex.quote(conda_prefix)}",
                # /opt/dsh/bin must be on PATH so dsh's `#!/usr/bin/env node`
                # shebang resolves the node binary shipped in the tool image.
                f"PATH={shlex.quote(conda_prefix + '/bin')}:/opt/dsh/bin:/opt/miniconda3/bin:$PATH",
            ]
        )
    else:
        env_assignments.append("PATH=/opt/dsh/bin:$PATH")

    argv = [dsh_bin, "--profile", "headless"]
    if tool_mode == "minimal":
        argv += ["--patch", minimal_patch_path]
        logger.info("deepseek_harness: tool_mode=minimal -> %s (bash + str_replace_editor)", minimal_patch_path)
    elif tool_mode != "full":
        raise ValueError(f"deepseek_harness: unsupported tool_mode={tool_mode!r} (expected 'full' or 'minimal')")

    # The task is the headless positional.
    argv.append(task)
    env_prefix = " ".join(env_assignments)
    return f"cd /testbed; mkdir -p {DSH_HOME_IN_SANDBOX}; {env_prefix} " + shlex.join(argv)


class DeepSeekHarnessConfig(AgentConfig):
    """Black-box launch params for DeepSeek Harness (endpoint lives on :attr:`AgentConfig.model`)."""

    name: str = "deepseek_harness"
    run_timeout: float = Field(default=7200.0, description="Wallclock cap (s) on the dsh process.")
    conda_env: str = Field(default="testbed", description="Task repo conda env, activated around the launch.")
    tool_mode: str = Field(
        default="minimal",
        description="'minimal' (bash + str_replace_editor via --patch, the default) or 'full' (25-tool Code Mode).",
    )
    # Tool-image paths are bound to the prebuilt tool image's Dockerfile layout
    # (mounted at /opt/dsh), declared here so the task config can override if
    # that layout changes.
    dsh_bin: str = Field(default="/opt/dsh/bin/dsh", description="dsh CLI inside the mounted tool image.")
    minimal_patch_path: str = Field(
        default="/opt/dsh/minimal_patch.yml",
        description="Patch overlay baked into the tool image; applied when tool_mode=minimal.",
    )


@register_agent("deepseek_harness")
class DeepSeekHarnessAgent(Agent):
    """Black-box solver: launch `dsh --profile headless` in the sandbox against ``config.model``."""

    config_model = DeepSeekHarnessConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        cfg: DeepSeekHarnessConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError(
                "deepseek_harness: config.model.base_url is not set (run_task fills the gateway tunnel address)"
            )
        task = self._extract_task(messages)

        # run_task has already rewritten cfg.model.base_url to the sandbox-internal
        # tunnel (http://127.0.0.1:<proxy_port>/v1), so it passes through as-is.
        agent_cmd = build_dsh_command(
            task=task,
            base_url=cfg.model.base_url,
            conda_env=cfg.conda_env,
            tool_mode=cfg.tool_mode,
            dsh_bin=cfg.dsh_bin,
            minimal_patch_path=cfg.minimal_patch_path,
        )
        # TODO(train): DSH headless has no CLI --max-turns flag (unlike Claude
        # Code's); a per-turn bound needs a DSH config/patch knob. For now the
        # wall-clock run_timeout is the only bound.
        result = await sandbox.exec_shell(agent_cmd, timeout=cfg.run_timeout)

        # DSH headless exits 0 when the agent finishes normally; treat that as a
        # finished episode (non-zero => crash/timeout => masked from loss via
        # mask_unfinished_episode=True in the framework config).
        finished = result.exit_code == 0
        logger.info(
            "deepseek_harness: done rc=%s tool_mode=%s finished=%s",
            result.exit_code,
            cfg.tool_mode,
            finished,
        )
        return AgentResult(
            output={
                "dsh_exit_code": result.exit_code,
                "stdout_tail": (result.stdout or "")[-2000:],
                "tool_mode": cfg.tool_mode,
            },
            transcript=list(messages),
            info={"tool_mode": cfg.tool_mode, "exit_code": result.exit_code},
            finished=finished,
        )

    @staticmethod
    def _extract_task(messages: list[dict[str, Any]]) -> str:
        if len(messages) > 2:
            raise ValueError(f"deepseek_harness accepts at most 2 messages (system?, user), got {len(messages)}")
        problem = next((m["content"] for m in messages if m.get("role") == "user"), None)
        if not problem:
            raise ValueError("deepseek_harness requires a 'user' message (the problem statement)")
        return problem
