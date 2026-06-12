# ── YR (openYuanrong) deployment ────────────────────────────────────────
# Connection credentials for the YR/AKernel sandbox platform.
# Set these before running, or provide them via agent_config YAML.
export DEPLOYMENT=openyuanrong
export OPENYUANRONG_SERVER_ADDRESS="${OPENYUANRONG_SERVER_ADDRESS:-REPLACE_ME}"

export OPENYUANRONG_TOKEN="${OPENYUANRONG_TOKEN:-REPLACE_ME}"

# Shell command prepended to swerex startup inside the sandbox.
# Can also be set in agent_config YAML via env.deployment.env_prepare_cmd.
export OPENYUANRONG_ENV_PREPARE_CMD="unset http_proxy && unset https_proxy && unset no_proxy && pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && pip config set install.trusted-host repo.huaweicloud.com && python3 -m pip install -q swe-rex"
