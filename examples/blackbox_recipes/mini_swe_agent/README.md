# Mini-SWE-Agent In-Sandbox Execution

## Overview

[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) runs inside the
SWE-bench sandbox through a sidecar tool image. The external runner creates the
sandbox, mounts the tool image at `/opt/mini-swe-agent`, pipes the task config
to `run_agent.py` via base64-encoded stdin, parses the result JSON from stdout,
and evaluates the reward in the same sandbox.

Unlike the [claude-code recipe](../claude_code), which builds a single
`claude -p ...` command and executes it directly, this recipe uses an
in-sandbox Python entrypoint (`run_agent.py`) that constructs the mini-swe-agent
`DefaultAgent` programmatically and points its `LitellmModel` at the gateway
through the sandbox-internal tunnel (`gateway_url` rewritten to
`http://127.0.0.1:<proxy_port>`).

The mini-swe-agent tool image uses
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
to build an isolated Python venv (independent of the sandbox base image's
glibc), installs `mini-swe-agent` + `litellm` into it, and copies the result
into a minimal `FROM scratch` final stage. The sandbox base image therefore does
not need Python for the sidecar tool runtime.

**This recipe is self-contained.** It builds the sandbox via the shared
[`uni_agent.sandbox`](../../uni_agent/sandbox) abstraction (`build_sandbox` +
`SandboxConfig`), selecting the provider through `SANDBOX_PROVIDER` (default
`openyuanrong`); everything else (`dataset.py`, `reward.py`, `build_tool.sh`,
`run_train.sh`, config) lives in this directory and does not depend on
`claude_code/`.

**Supported runners:**

| runner | Description |
|--------|-------------|
| `mini_swe_agent` | mini-swe-agent sidecar runner (stdin-pipe + run_agent.py) |

**Supported sandbox types:**

| Type | Description |
|------|-------------|
| openyuanrong | Uses `openyuanrong_sandbox_sdk.Mount` and `sandbox.exec_shell()` |

## Architecture

```text
[Rollouter Host: mini_swe_agent_runner]
  |
  |-- build_sandbox(SandboxConfig(provider=SANDBOX_PROVIDER, image, mounts=[{target="/opt/mini-swe-agent", image_url=sidecar}], upstream=..., proxy_port=...))
  |     `-- provider Sandbox (e.g. openYuanrong: Sandbox(mounts=[Mount(target="/opt/mini-swe-agent", ...)]))
  |
  |-- sandbox.exec_shell("printf <config_b64> | base64 -d | /opt/mini-swe-agent/bin/python /opt/mini-swe-agent/bin/run_agent.py")
  |     `-- [Inside Sandbox]
  |           run_agent.py reads {task, gateway_url, agent.step_limit} from stdin
  |           LitellmModel(api_base=gateway_url -> 127.0.0.1:<proxy_port>) runs DefaultAgent
  |           commands run inside the SWE-bench sandbox /testbed
  |           result JSON written to stdout: {exit_status, submission, model_stats}
  |
  |-- _parse_agent_result(stdout) -> {exit_status, submission}
  |-- SandboxEnvForReward(sandbox) -> evaluate_in_env()  (uni_agent.tasks.swe_bench.reward.compute_reward)
  `-- POST session.reward_info_url
```

## stdin/stdout contract with `run_agent.py`

The runner pipes a base64-encoded JSON task config to `run_agent.py`'s stdin:

```json
{
  "task": "<issue description for the agent to solve>",
  "gateway_url": "http://127.0.0.1:38197/v1",
  "agent": {"step_limit": 100}
}
```

`run_agent.py` writes a single JSON line to stdout on completion:

```json
{
  "exit_status": "submitted" | "error" | "<ExceptionClassName>",
  "submission": "<git diff>",
  "model_stats": {"instance_cost": 0.0, "api_calls": 0}
}
```

The runner parses the last stdout line starting with `{` (litellm may print
noise to stdout) via `_parse_agent_result`.

## Prerequisites

1. **OpenYuanrong** - set `OPENYUANRONG_SERVER_ADDRESS` and `OPENYUANRONG_TOKEN`.
2. **Tool image** — build the mini-swe-agent tool image and push it to a remote
   registry if the sandbox service cannot access local Docker images:
   ```bash
   bash examples/blackbox_recipes/mini_swe_agent/build_tool.sh \
       --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
   ```
3. **SWE-bench data** — a parquet with per-sample `extra_info.tools_kwargs`
   (env image + reward metadata). See
   [`examples/data_preprocess/`](../../data_preprocess/) for dataset prep.

## Running

### Standalone inference (no trainer)

```bash
bash examples/blackbox_recipes/mini_swe_agent/run_infer.sh
```

Reports a resolve rate over the dataset. Override via env vars, e.g.
`MODEL_PATH=... DATA_PATH=... N=4 MAX_SAMPLES=10 bash .../run_infer.sh`.

### RL training (Megatron + V1 async)

```bash
bash examples/blackbox_recipes/mini_swe_agent/run_train.sh
```

Submits a `ray job` running `verl.trainer.main_ppo` with the V1 unified trainer
in `separate_async` mode (default: 4 trainer GPUs + 4 rollout GPUs on one
8-GPU node). See the script header for the full env-var surface
(`TRAINER_MODE`, `GEN_TP`, `N`, `AGENT_MAX_TURNS`, `MINI_SWE_PROXY_PORT`, ...).

## Tool image internals

`Dockerfile.mini-swe-agent-tool` builds two stages:

1. **builder** (`debian:bullseye-slim`): downloads python-build-standalone
   (stripped, ~32 MB), extracts it to `/opt/mini-swe-agent`, and `pip install`s
   `mini-swe-agent==2.2.8` + `litellm==1.81.7` into that venv. The in-sandbox
   runner `run_agent.py` is copied to `/opt/mini-swe-agent/bin/`.
2. **final** (`FROM scratch`): copies `/opt/mini-swe-agent` to the image root so
   that `mount(target="/opt/mini-swe-agent")` overlays the files at
   `/opt/mini-swe-agent/bin/python` etc. in the sandbox.

Override the python-build-standalone release / pip index at build time:

```bash
docker build -f Dockerfile.mini-swe-agent-tool \
    --build-arg PBS_RELEASE=20260602 \
    --build-arg PBS_PYTHON=3.12.13 \
    --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ \
    -t mini-swe-agent-tool:latest .
```
