alias wnpu="watch -n 0.1 'npu-smi info | tail -n 44'"
# ── NPU / HCCL ──────────────────────────────────────────────────────────
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-16666-17000}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"

# ── K8s / 集群 ──────────────────────────────────────────────────────────
export KUBECONFIG="${KUBECONFIG:-kubeconfig.yaml}"

# ── Ray ──────────────────────────────────────────────────────────────────
export RAY_memory_monitor_refresh_ms=0
export RAY_raylet_heartbeat_timeout_milliseconds=600000
export RAY_num_heartbeats_timeout=1000

# ── vLLM ────────────────────────────────────────────────────────────────
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65535}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65535}"

# ── 路径 ────────────────────────────────────────────────────────────────
export PATH="${NYDUS_PATH:-nydus-static}:$PATH"
export HF_ENDPOINT=https://hf-mirror.com

# ── YR (openYuanrong) deployment ────────────────────────────────────────
export DEPLOYMENT=openyuanrong
export OPENYUANRONG_SERVER_ADDRESS="${OPENYUANRONG_SERVER_ADDRESS:-REPLACE_ME}"
export AKERNEL_SERVER_ADDRESS="${AKERNEL_SERVER_ADDRESS:-REPLACE_ME}"

export OPENYUANRONG_TOKEN="${OPENYUANRONG_TOKEN:-REPLACE_ME}"
export AKERNEL_TOKEN="${AKERNEL_TOKEN:-REPLACE_ME}"
# _swerex_start_command 会将其前置到 swerex 启动命令
export OPENYUANRONG_ENV_PREPARE_CMD="unset http_proxy && unset https_proxy && unset no_proxy && pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && pip config set install.trusted-host repo.huaweicloud.com && python3 -m pip install -q swe-rex"