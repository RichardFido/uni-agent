# Mini-SWE-Agent In-Sandbox Execution

## Overview

`mini-swe-agent` runs inside the SWE-bench sandbox through a sidecar tool image.
The runner creates the sandbox, mounts the tool image at `/opt/mini-swe-agent`,
pipes a task config JSON to the in-sandbox `run_agent.py` via **stdin**, and
reads the agent result JSON from **stdout**. The reward is then evaluated in the
same sandbox by the task's own reward function.

The agent executes commands through mini-swe-agent's `LocalEnvironment` (local
bash) inside the sandbox and calls the LLM through the gateway URL passed in via
stdin. The tool image uses
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
to build an isolated Python environment, then copies the result into a minimal
`FROM scratch` final stage, so the sandbox base image does not need to provide
Python for the sidecar tool runtime.

**This recipe is a first-class `uni_agent` agent.** The agent lives in
`uni_agent/agents/mini_swe_agent/` and the sandbox/reward reuse the framework's
`openyuanrong` provider and `swe_rebench`/`swe_bench` tasks. Training wires the
unified runner bridge `uni_agent.framework.task_runner.run_task`, which resolves
each sample's task from `task_config_mini_swe_agent.yaml` (agent + sandbox
defaults) and deep-merges the runtime model binding.

**Supported runners:**

| runner | Description |
|--------|-------------|
| `mini_swe_agent` | mini-swe-agent sidecar runner (`uni_agent.agents.mini_swe_agent`) |

**Supported sandbox types:**

| Type | Description |
|------|-------------|
| openyuanrong | `uni_agent.sandbox.openyuanrong` (canonical SWE image refs are prefixed with the openyuanrong registry) |

## Architecture

```text
[Rollouter: run_task (uni_agent.framework.task_runner)]
  |
  |-- TaskConfigResolver: task_config YAML + sample row + runtime model
  |     `-- sandbox: provider=openyuanrong, image=<canonical>, mounts=[tool image -> /opt/mini-swe-agent]
  |     `-- agent:   MiniSweAgentAgent (step_limit/run_timeout/conda_env/proxy_port)
  |
  |-- [run_task tunnel injection] sandbox_kwargs.upstream = gateway host:port (runtime)
  |
  |-- OpenyuanrongSandbox.start() (mounts + upstream/proxy_port reverse tunnel)
  |-- MiniSweAgentAgent.run()
  |     `-- sandbox.exec_shell("printf %s <b64> | base64 -d | env <conda> /opt/mini-swe-agent/bin/python /opt/mini-swe-agent/bin/run_agent.py")
  |           stdin <- task config JSON {task, gateway_url: http://127.0.0.1:<proxy_port>/v1, agent:{step_limit}}
  |           stdout -> agent result JSON {exit_status, submission, model_stats}
  `-- compute_reward(metadata, sandbox) -> reward_info POST via report_reward=True
```

## Prerequisites

1. **OpenYuanrong** — set `OPENYUANRONG_SERVER_ADDRESS` and `OPENYUANRONG_TOKEN`.
2. **Tool image** — build the mini-swe-agent tool image and push it to a remote
   registry if the sandbox service cannot access local Docker images.

## 1. Build Tool Image

`mini_swe_agent` is injected into the SWE-bench sandbox as a sidecar tool image.
Use `build_tool.sh` to build it.

| Default tool image | Dockerfile | Sandbox mount path | Image contents |
|--------------------|------------|--------------------|----------------|
| `mini-swe-agent-tool:latest` | `Dockerfile.mini-swe-agent-tool` | `/opt/mini-swe-agent` | Standalone Python 3.12, `mini-swe-agent`, `litellm`, and `run_agent.py` |

```bash
# Use the default PyPI source.
bash examples/blackbox_recipes/mini_swe_agent/build_tool.sh

# Use a custom PyPI mirror.
bash examples/blackbox_recipes/mini_swe_agent/build_tool.sh --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/

# Build and push to a remote registry.
bash examples/blackbox_recipes/mini_swe_agent/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

The tool image URL is referenced from `task_config_mini_swe_agent.yaml`
(`sandbox.sandbox_kwargs.mounts[].image_url`), not from the training script.

## 2. Data Preparation

Re-run the new preprocess to produce parquet rows whose `extra_info.tools_kwargs.task`
carries `{name, sandbox:{image: canonical}, prompt, metadata}` for `run_task`:

```bash
python -m uni_agent.tasks.swe_rebench.preprocess --local-save-dir ~/data/uni_agent
python -m uni_agent.tasks.swe_bench.preprocess     --local-save-dir ~/data/uni_agent
```

## 3. Training (Fully Async)

```bash
OPENYUANRONG_SERVER_ADDRESS="<server-address>" \
OPENYUANRONG_TOKEN="<token>" \
MODEL_PATH=~/models/Qwen3.5-9B \
TRAIN_DATA=~/data/uni_agent/swe_rebench_filtered.parquet \
VAL_DATA=~/data/uni_agent/swe_bench_verified.parquet \
bash examples/blackbox_recipes/mini_swe_agent/run_train.sh
```

The runner is fixed to the unified bridge:

```text
agent_runners.task.runner_fqn = uni_agent.framework.task_runner.run_task
```

## 4. Configuration

### Task config (agent + sandbox)

Most recipe knobs are configured in `task_config_mini_swe_agent.yaml`; edit it to
tune the agent without touching the training script:

| Key | Default | Description |
|-----|---------|-------------|
| `sandbox.sandbox_kwargs.mounts[].image_url` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest` | Sidecar tool image |
| `sandbox.sandbox_kwargs.proxy_port` | `38197` | Sandbox-internal reverse tunnel port (single source of truth) |
| `agent.step_limit` | `100` | mini-swe-agent max agent steps |
| `agent.run_timeout` | `7200` | Max wall time for the agent process in the sandbox |
| `agent.conda_env` | `testbed` | Conda env activated inside the sandbox before running the agent |

> `sandbox.sandbox_kwargs.upstream` and `agent.model.base_url/api_key/model_name`
> are **runtime-derived** (`session.base_url`) and injected by `run_task`; do not
> set them in the YAML. `agent.proxy_port` is kept in sync with the sandbox's
> `proxy_port` automatically.

### Training script env vars (selected)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENYUANRONG_SERVER_ADDRESS` / `OPENYUANRONG_TOKEN` | — | **Required** openyuanrong sandbox credentials |
| `OPENYUANRONG_IMAGE_REGISTRY` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong` | Override the canonical→registry image prefix |
| `SANDBOX_NAME_PREFIX` | `mini-swe-` | Prefix for created sandbox names |
| `TASK_CONFIG` | `examples/blackbox_recipes/mini_swe_agent/task_config_mini_swe_agent.yaml` | Task-config YAML |
| `GATEWAY_COUNT` | `4` | Gateway actors fronting the engine |
| `MAX_CONCURRENT_SESSIONS` | `128` | Max in-flight rollout sessions (runner cap) |
| `MASK_UNFINISHED_EPISODE` | `False` | Set `True` to zero the loss mask for unfinished episodes |
| `SERVED_MODEL_NAME` | `basename ${MODEL_PATH}` | Model name served at the gateway |
| `TOOL_PARSER` | `qwen3_coder` | Gateway tool-call parser (must match the model chat template) |
| `MODEL_PATH` / `TRAIN_DATA` / `VAL_DATA` | `~/models/Qwen3.5-9B` etc. | Model + dataset paths |

`MASK_UNFINISHED_EPISODE=True` is meaningful now: the agent reports `finished`
explicitly (`exit_status == "Submitted"`), so errored/timed-out episodes can be
excluded from the loss instead of trained on a zero reward.
