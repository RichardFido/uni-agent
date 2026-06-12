# ── NPU / HCCL ──────────────────────────────────────────────────────────
alias wnpu="watch -n 0.1 'npu-smi info | tail -n 44'"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-16666-17000}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"

# ── Ray ──────────────────────────────────────────────────────────────────
export RAY_memory_monitor_refresh_ms=0
export RAY_raylet_heartbeat_timeout_milliseconds=600000
export RAY_num_heartbeats_timeout=1000

# ── vLLM (NPU-specific timeouts) ────────────────────────────────────────
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
