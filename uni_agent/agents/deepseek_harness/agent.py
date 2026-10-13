"""deepseek_harness: a black-box agent that runs the real `dsh --profile headless`
inside the sandbox.

DeepSeek Harness is launched *in* the sandbox from a prebuilt tool image mounted
at ``/opt/dsh`` (Node CLI + node runtime + the baked ``minimal_patch.yml``).
This agent builds the ``dsh --profile headless`` shell command, points DSH's
built-in ``llm-deepseek`` route at the gateway tunnel, execs it in the sandbox,
and returns the result.

Unlike ``mini_swe_agent`` there is **no in-sandbox ``run_agent.py`` entrypoint**:
``dsh`` is itself the agent binary, so this agent builds and runs the command
directly. The unified ``uni_agent.framework.task_runner.run_task`` bridge owns the
gateway reverse tunnel (``upstream`` + ``model.base_url`` rewrite to
``http://127.0.0.1:<proxy_port>/v1``) and the reward, so this agent only owns the
launch.

Wire compat: DSH speaks OpenAI Chat Completions streaming with tool_calls, so it
uses the gateway's OpenAI adapter directly.
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
# must be writable inside the sandbox. Auto-init uses the templates shipped in
# the tool image, so it needs no network.
DSH_HOME_IN_SANDBOX = "/tmp/dsh-home"


def build_dsh_command(
    *,
    task: str,
    base_url: str,
    conda_env: str = "testbed",
    tool_mode: str = "full",
    dsh_bin: str = "/opt/dsh/bin/dsh",
    minimal_patch_path: str = "/opt/dsh/minimal_patch.yml",
    workdir: str | None = None,
    model_name: str | None = None,
    max_model_len: int | None = None,
    permission_mode: str = "danger-full-access",
    step_limit: int | None = None,
    disable_title_generation: bool = True,
    disable_auto_compaction: bool = True,
) -> str:
    """Build the shell command that runs DSH headless against the gateway tunnel.

    ``base_url`` is already the sandbox-internal tunnel address
    (``http://127.0.0.1:<proxy_port>/v1``) -- ``run_task`` rewrites
    ``agent.model.base_url`` before this agent runs, so the agent is
    tunnel-agnostic.

    ``model_name`` / ``max_model_len`` (optional): when the endpoint is an
    OpenAI-compatible server that serves ``model_name`` (e.g. vLLM) rather than
    DeepSeek's official ``deepseek-v4-flash``, DSH defaults to 256k output
    tokens and its default catalog model, both of which fail (404 unknown model,
    max_tokens > max_model_len). Supplying these generates a runtime
    ``llm-deepseek`` patch that points ``agent-default-model`` at ``model_name``
    and caps ``maxTokens`` below ``max_model_len``.

    ``tool_mode``:
      * ``"full"``    -- the default 25-tool Code Mode surface.
      * ``"minimal"`` -- apply the baked ``minimal_patch.yml`` to restrict the
        request to ``bash`` + ``str_replace_editor`` (measured: 25 -> 2 tools,
        4122 -> 257 sys_chars).
    """
    # DSH's llm-deepseek route honors DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY. The
    # gateway has no auth path, so a dummy key is fine.
    env = {
        "DEEPSEEK_BASE_URL": base_url,
        "DEEPSEEK_API_KEY": "not-needed",
        "DSH_HOME": DSH_HOME_IN_SANDBOX,
        "IS_SANDBOX": "1",
        "DISABLE_AUTOUPDATER": "1",
        # Inside the openYuanrong outer sandbox DSH has no inner bash-sandbox
        # backend (bubblewrap/Landlock/approval channel); default to
        # danger-full-access so DSH's bash tool runs against the already-isolated
        # /testbed. Override per deployment via cfg.permission_mode.
        "DSH_PERMISSION_MODE": permission_mode,
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

    # Optional overlay that points DSH's llm-deepseek route at the served model
    # with a maxTokens cap compatible with the endpoint's max_model_len.
    # Written as a standalone command BEFORE the env-prefixed dsh command:
    # a heredoc delimiter must end its own line, so it cannot be spliced into
    # the space-joined env prefix (that would swallow the dsh argv into the
    # heredoc body and never run dsh). Each building block is its own line in
    # the produced shell script (no leading ';' on any line).
    blocks = [f"cd {shlex.quote(workdir or '/testbed')}; mkdir -p {DSH_HOME_IN_SANDBOX}"]
    if model_name:
        model_patch_path, model_patch_script = _build_model_overlay_patch(
            model_name=model_name,
            max_model_len=max_model_len,
            path="/tmp/dsh_model_patch.yml",
        )
        argv += ["--patch", model_patch_path]
        blocks.insert(0, model_patch_script)

    # Optional step watchdog: cap agent steps by counting DSH's plaintext session
    # log. DSH has no native step flag, so we (1) force the session-persistence
    # plugin to write uncompressed JSONL via a patch, then (2) run dsh in the
    # background under a watchdog that kills it once `step/start` events reach
    # `step_limit`. A killed dsh exits non-zero -> finished=False -> masked from
    # the loss (same semantics as mini_swe_agent.step_limit).
    use_watchdog = step_limit is not None and step_limit > 0
    if use_watchdog:
        session_patch_path, session_patch_script = _build_session_log_patch(
            path="/tmp/dsh_session_patch.yml",
        )
        argv += ["--patch", session_patch_path]
        blocks.append(session_patch_script)
        blocks.append(_build_step_watchdog(step_limit=step_limit, session_dir=DSH_HOME_IN_SANDBOX))

    # Optional data-hygiene patch: disable DSH's auto session-title and auto
    # context-compaction so each rollout yields a single clean task trajectory
    # (no 1-turn title row, no compact-boundary split). Written under /tmp so it
    # does not depend on the mounted tool image being writable.
    hygiene_patch_path, hygiene_patch_script = _build_data_hygiene_patch(
        path="/tmp/dsh_data_hygiene.yml",
        disable_title=disable_title_generation,
        disable_auto_compaction=disable_auto_compaction,
    )
    if hygiene_patch_path:
        argv += ["--patch", hygiene_patch_path]
        blocks.append(hygiene_patch_script)

    # The task is the headless positional.
    argv.append(task)
    env_prefix = " ".join(env_assignments)
    dsh_run = f"{env_prefix} " + shlex.join(argv)
    if use_watchdog:
        dsh_run = f"{dsh_run} &\nDSPID=$!\n" + _watchdog_wait_block()
    return "\n".join(blocks) + f"\n{dsh_run}"


def _build_model_overlay_patch(
    *,
    model_name: str,
    max_model_len: int | None,
    path: str,
) -> tuple[str, str]:
    """Return ``(patch_path, shell_script)`` writing a DSH llm-deepseek overlay patch.

    DSH's ``llm-deepseek`` defaults to ``maxTokens=256e3`` and a catalog of
    official DeepSeek models; neither works against a vLLM / OpenAI-compatible
    endpoint serving ``model_name`` with a smaller context. The patch:
    - replaces the catalog ``models`` with the served model (``contextWindow``
      and ``maxTokens`` capped under ``max_model_len`` when known), and
    - points ``agent-default-model`` at ``model_name``.

    The returned script is appended to the command's env prefix so it runs
    before dsh boots (it uses ``bash -c`-style redirection, safe to embed).
    """
    max_tokens = None
    if max_model_len is not None and max_model_len > 0:
        # Leave headroom for the prompt inside the endpoint's total context.
        max_tokens = int(max_model_len * 0.8)
    lines = ["- id: llm-deepseek", "  config:"]
    if max_tokens is not None:
        lines += [f"    maxTokens: {max_tokens}"]
    lines += ["    models:"]
    lines += [
        f"      - id: {model_name}",
        f"        name: {model_name}",
        f"        contextWindow: {max_model_len or 131072}",
    ]
    if max_tokens is not None:
        lines += [f"        maxTokens: {max_tokens}"]
    lines += [
        "- id: agent-default-model",
        "  config:",
        "    provider: deepseek-official",
        f"    model: {model_name}",
        "",
    ]
    body = "\n".join(lines)
    # The heredoc delimiter must terminate on its own line; the trailing
    # newline lets the caller append "; ..." on the next line.
    script = (
        f"cat > {shlex.quote(path)} <<'DSH_MODEL_PATCH_EOF'\n{body}"
        "DSH_MODEL_PATCH_EOF\n"
    )
    logger.info("deepseek_harness: model overlay patch (model_name=%s max_model_len=%s)", model_name, max_model_len)
    return path, script


def _build_data_hygiene_patch(*, path: str, disable_title: bool, disable_auto_compaction: bool) -> tuple[str, str]:
    """Return ``(patch_path, shell_script)`` writing a DSH patch quality switch.

    Two optional switches kept out of ``minimal_patch.yml`` so they can be
    toggled per-deployment without rebuilding/overriding the mounted tool image:

    * ``disable_title`` -- disables ``session-title`` and ``session-title-llm``.
      DSH auto-invokes a lightweight session-title generation on the first
      prompt (a ~64-token LLM call). Inside the training gateway that call is
      materialized as an extra 1-turn trajectory that carries the *task's*
      reward but no task signal, polluting the batch (~45% of trajectory
      objects in a step are title rows). Disabling it removes those rows.
    * ``disable_auto_compaction`` -- sets ``compaction-basic.auto=false``,
      turning off both the step-pressure compaction (contextWindow *
      thresholdRatio trigger) and the context-overflow auto-recovery. Effect:
      no compact-split trajectory segments in training data; cost: tasks that
      exceed ``contextWindow`` fail with a context-window-exceeded error and
      are masked from the loss via ``mask_unfinished_episode=True``.

    The generated file is written under the sandbox ``/tmp`` (writable) and
    appended to the ``dsh --patch`` chain, so it does not depend on the mounted
    tool image being writable.
    """
    body = []
    if disable_title:
        body += [
            "- id: session-title",
            "  disabled: true",
            "- id: session-title-llm",
            "  disabled: true",
        ]
    if disable_auto_compaction:
        body += [
            "- id: compaction-basic",
            "  config:",
            "    auto: false",
        ]
    if not body:
        return "", ""
    body.append("")
    script = (
        f"cat > {shlex.quote(path)} <<'DSH_HYGIENE_PATCH_EOF'\n{chr(10).join(body)}"
        "DSH_HYGIENE_PATCH_EOF\n"
    )
    logger.info("deepseek_harness: data-hygiene patch (disable_title=%s disable_auto_compaction=%s)", disable_title, disable_auto_compaction)
    return path, script


def _build_session_log_patch(*, path: str) -> tuple[str, str]:
    """Return ``(patch_path, shell_script)`` forcing DSH's session log to plaintext.

    ``dsh-session-persistence-jsonl`` defaults to ``compression: zstd`` (writes
    ``session.jsonl.zstd``). The step watchdog needs to count ``step/start``
    events cheaply, so this overlay sets ``compression: none`` -> plaintext
    ``session.jsonl`` under ``$DSH_HOME/sessions/...``.
    """
    body = "\n".join(
        [
            "- id: session-persistence-jsonl",
            "  config:",
            "    root: !!js dshHomePath('sessions')",
            "    compression: none",
            "",
        ]
    )
    script = (
        f"cat > {shlex.quote(path)} <<'DSH_SESSION_PATCH_EOF'\n{body}"
        "DSH_SESSION_PATCH_EOF\n"
    )
    logger.info("deepseek_harness: session-persistence patched to plaintext JSONL (for step watchdog)")
    return path, script


def _build_step_watchdog(*, step_limit: int, session_dir: str) -> str:
    """Return a shell snippet that kills a background dsh after ``step_limit`` steps.

    Counts ``"type":"step/start"`` lines across every plaintext session log under
    ``session_dir`` (``$DSH_HOME/sessions``), every 0.5s, and once the count
    reaches ``step_limit`` kills the dsh binary process (scoped via its absolute
    path so unrelated processes are untouched).
    """
    sessions_glob = f"{shlex.quote(session_dir)}/sessions/*/*/session.jsonl"
    return (
        "(\n"
        "  while :; do\n"
        f"    N=$(grep -ah '\"type\":\"step/start\"' {sessions_glob} 2>/dev/null | wc -l)\n"
        f"    if [ \"${{N:-0}}\" -ge {int(step_limit)} ]; then\n"
        f"      pkill -9 -f {shlex.quote('/opt/dsh/bin/dsh')} 2>/dev/null\n"
        "      break\n"
        "    fi\n"
        "    sleep 0.5\n"
        "  done\n"
        ") &\n"
    )


def _watchdog_wait_block() -> str:
    """Return the shell closing snippet that waits on the backgrounded dsh.

    ``build_dsh_command`` backgrounds dsh as ``$DSPID`` and runs the watchdog in
    a subshell; this block waits for ``$DSPID`` to finish (either normally or by
    the watchdog's kill) and lets the outer script exit with dsh's real status.
    """
    return (
        "wait $DSPID\n"
        "RC=$?\n"
        "exit $RC\n"
    )


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
    permission_mode: str = Field(
        default="danger-full-access",
        description="``DSH_PERMISSION_MODE`` for the headless run. Inside an outer isolation "
        "sandbox (openYuanrong), DSH's own inner bash sandbox has no backend (no bubblewrap/"
        "Landlock/approval channel), so ``bash`` tool calls fail unless set to "
        "``danger-full-access`` (letting DSH run unconfined because the outer sandbox already "
        "isolates it). Set ``workspace-write``/``read-only`` only when DSH runs directly on a host.",
    )
    max_model_len: int | None = Field(
        default=None,
        description="Endpoint max_model_len (tokens). When set alongside agent.model.model_name, "
        "dsh's llm-deepseek route is pointed at that served model with maxTokens capped under "
        "this context (required when the endpoint is vLLM/OpenAI-compatible rather than DeepSeek "
        "official, whose default catalog model / 256k max_tokens fail).",
    )
    step_limit: int | None = Field(
        default=None,
        description="Max agent steps (DSh 'step' boundaries) before the episode is cut short. "
        "DSH has no native --max-steps flag, so a background watchdog inside the launch command "
        "counts ``step/start`` events in DSH's plaintext session log and kills the dsh process "
        "once the limit is reached. The killed dsh exits non-zero, so ``finished=False`` and the "
        "episode is masked from the loss via ``mask_unfinished_episode=True`` (same training "
        "semantics as mini_swe_agent.step_limit). None = no step cap (only run_timeout).",
    )
    disable_title_generation: bool = Field(
        default=True,
        description="Disable DSH's auto session-title generation (first-prompt LLM title call). "
        "Inside the training gateway that call materializes as a ~64-token extra trajectory on "
        "every rollout (0-indexed, carrying the task reward but no task signal). Default True to "
        "keep training batches clean.",
    )
    disable_auto_compaction: bool = Field(
        default=True,
        description="Disable DSH's auto context compaction (compaction-basic.auto=false). "
        "Stops step-pressure compaction and context-overflow auto-recovery, so a rollout never "
        "splits into compact-boundary segments. Tasks exceeding contextWindow will fail with a "
        "context-window-exceeded error and be masked from the loss. Default True (prefer clean "
        "single-trajectory rollouts over recovering over-length tasks).",
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
        workdir: str | None = None,
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
            workdir=workdir or "/testbed",
            model_name=cfg.model.model_name,
            max_model_len=cfg.max_model_len,
            permission_mode=cfg.permission_mode,
            step_limit=cfg.step_limit,
            disable_title_generation=cfg.disable_title_generation,
            disable_auto_compaction=cfg.disable_auto_compaction,
        )
        # DSH headless has no CLI --max-turns flag; step_limit (when set) is
        # enforced in-sandbox by a step-counting watchdog inside the launch
        # command (kills dsh at the limit -> non-zero exit -> finished=False).
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
                "step_limit": cfg.step_limit,
            },
            transcript=list(messages),
            info={
                "tool_mode": cfg.tool_mode,
                "exit_code": result.exit_code,
                "step_limit": cfg.step_limit,
            },
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
