"""SWE-rebench task (native framework loop).

Same shape as :mod:`uni_agent.tasks.swe_bench.task`, with two swe-rebench specifics:
scoring reads the eval config carried on the row (see :mod:`.reward`), and the
future git history is cleaned in-sandbox before the agent runs (this used to be a
data-preprocess ``post_setup_cmd``; owning it here keeps the dataset row declarative).
"""

from __future__ import annotations

import json
import logging

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)

def git_clean_history(base_commit: str) -> str:
    """Return the shell script that cuts future (answer) history out of /testbed.

    The script moves main/HEAD onto ``base_commit`` so future (answer) commits
    are unreachable, then deletes every other branch/remote/tag ref and prunes
    unreachable objects. This must be a function because the exact commands
    depend on the per-sample ``base_commit`` (it lives on the dataset row, so it
    is not available at module import time). Best-effort (``|| true``); runs
    once in /testbed.
    """
    return " && ".join(
        [
            # Move main/HEAD onto base_commit so future (answer) commits are cut
            # out of the reachable history. Otherwise git log --all / git show can
            # still reach the reference fix through the main-branch ancestor chain
            # or leftover release tags even after tag-deletion + gc.
            f"git -C /testbed checkout -q {base_commit}",
            f"git -C /testbed branch -q -f main {base_commit}",
            f"git -C /testbed symbolic-ref HEAD refs/heads/main",
            # Delete every other branch/remote/tag ref that could reference future commits.
            "git -C /testbed for-each-ref --format='%(refname)' refs/heads/ refs/remotes/ refs/tags/ | "
            "grep -v '^refs/heads/main$' | while read ref; do git -C /testbed update-ref -d \"$ref\"; done",
            "git -C /testbed reflog expire --expire=now --all || true",
            "git -C /testbed gc --prune=now || true",
        ]
    )


class SWEREBenchTaskConfig(TaskConfig):
    name: str = "swe_rebench"
    run_oracle_solution: bool = Field(
        default=False,
        description="Oracle mode: skip the agent and score the dataset's gold patch directly.",
    )


@register_task("swe_rebench")
class SWEREBenchTask(Task):
    name = "swe_rebench"
    config_model = SWEREBenchTaskConfig

    async def run(self) -> TaskResult:
        cfg: SWEREBenchTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata  # the dataset sample is carried on the task config

        instance_id = sample.get("instance_id", "?") if isinstance(sample, dict) else "?"
        task_config_dump = cfg.model_dump(mode="json", exclude={"metadata", "prompt"})
        logger.info(
            f"starting swe_rebench task (instance_id={instance_id}, run_oracle_solution={cfg.run_oracle_solution})\n"
            f"task config: {json.dumps(task_config_dump, indent=2)}"
        )
        async with self.build_sandbox() as sandbox:
            # Clean future history before anything reads the repo.
            base_commit = sample["base_commit"]
            await sandbox.exec_shell(git_clean_history(base_commit), workdir="/testbed")

            if cfg.run_oracle_solution:
                logger.info("applying gold patch to /testbed")
                await sandbox.write_file("/tmp/gold_patch.patch", sample["patch"])
                await sandbox.exec(["git", "apply", "--whitespace=fix", "/tmp/gold_patch.patch"], workdir="/testbed")
                finished = True
            else:
                agent = self.build_agent()
                messages = cfg.prompt
                # The endpoint the agent calls lives on cfg.agent.model (the agent validates it).
                agent_result = await agent.run(sandbox=sandbox, messages=messages)
                finished = agent_result.finished

            try:
                from .reward import compute_reward

                result = await compute_reward(sample, sandbox)
            except Exception:
                logger.exception(f"scoring failed for instance_id={instance_id}")
                raise

            logger.info(f"task done: resolved={result['resolved']}")
            return TaskResult(
                reward=float(result["resolved"]),
                accuracy=float(result["resolved"]),
                finished=finished,
                extra_info=result,
            )
