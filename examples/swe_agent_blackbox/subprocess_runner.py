"""Ray-based subprocess runner for agent_runner execution.

Launches agent_runner in a separate Ray worker process to prevent blocking
operations (sleep, sync I/O, etc.) from stalling the framework's event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ray

from uni_agent.trainer.framework.types import SessionHandle

logger = logging.getLogger(__name__)


class _StubSessionRuntime:
    """Captures reward_info from agent_runner's complete_session call."""

    def __init__(self):
        self.reward_info: dict[str, Any] | None = None

    async def complete_session(self, session_id: str, reward_info: dict[str, Any] | None = None):
        self.reward_info = reward_info


@ray.remote(num_cpus=0)
def remote_agent_run(
    agent_runner_fqn: str,
    raw_prompt,
    session_id: str,
    base_url: str,
    sample_index: int,
    runner_kwargs: dict,
) -> dict[str, Any] | None:
    """Run agent_runner in a dedicated Ray worker process."""
    from verl.utils.import_utils import load_class_from_fqn

    agent_runner = load_class_from_fqn(agent_runner_fqn)
    stub_runtime = _StubSessionRuntime()
    handle = SessionHandle(session_id=session_id, base_url=base_url)

    async def _run():
        try:
            await agent_runner(
                raw_prompt=raw_prompt,
                session=handle,
                sample_index=sample_index,
                session_runtime=stub_runtime,
                **runner_kwargs,
            )
            return stub_runtime.reward_info
        except Exception as e:
            logger.error("remote_agent_run failed: session_id=%s, sample=%d, error=%s",
                         session_id, sample_index, e, exc_info=True)
            raise

    return asyncio.run(_run())
