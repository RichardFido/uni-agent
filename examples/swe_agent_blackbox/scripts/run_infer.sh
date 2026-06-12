#!/usr/bin/env bash
# Parallel inference for the blackbox SWE-agent recipe.
#
# Usage:
#   bash examples/swe_agent_blackbox/scripts/run_infer.sh

set -euo pipefail

# ── Model & data ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.5-9B}"
DATA_PATH="${DATA_PATH:-$HOME/data/swe_agent/swe_bench_verified_vefaas.parquet}"

# ── vLLM server parameters ─────────────────────────────────
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_MAX_NUM_SEQS=128
export VLLM_MAX_MODEL_LEN=65536
export VLLM_MAX_NUM_BATCHED_TOKENS=65536

# ── Inference parameters ─────────────────────────────────────────────────
MAX_SAMPLES="${MAX_SAMPLES:-500}"
PROMPT_LENGTH="${PROMPT_LENGTH:-16384}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-65536}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.7}"
N="${N:-1}"
ENGINE="${ENGINE:-vllm}"
TP="${TP:-2}"

# ── Agent parameters ─────────────────────────────────────────────────────
RUNNER="${RUNNER:-uniagent}"
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-examples/swe_agent_blackbox/config/agent_config.yaml}"
export SWE_AGENT_MAX_TURNS="${SWE_AGENT_MAX_TURNS:-100}"
export SWE_AGENT_ACTION_TIMEOUT="${SWE_AGENT_ACTION_TIMEOUT:-600}"
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"

# ── Logging ──────────────────────────────────────────────────────────────
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-DEBUG}"
export SWE_AGENT_LOG_TRAJECTORY=1
export DEBUG_MODE=True

# ── HCCL port range (avoid port conflict with multiple engine instances) ──
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-16666-17000}"

echo "=== SWE-Agent Blackbox Inference ==="
echo "Model: ${MODEL_PATH}"
echo "Data:  ${DATA_PATH}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Engine: ${ENGINE} (TP=${TP})"
echo "====================================="

python examples/swe_agent_blackbox/parallel_infer.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --max-samples "${MAX_SAMPLES}" \
    --prompt-length "${PROMPT_LENGTH}" \
    --response-length "${RESPONSE_LENGTH}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --n "${N}" \
    --engine "${ENGINE}" \
    --tensor-parallel-size "${TP}" \
    --max-turns "${SWE_AGENT_MAX_TURNS}" \
    --runner "${RUNNER}" \
    --agent-config-path "${AGENT_CONFIG_PATH}" \
    --n-gpus-per-node 8