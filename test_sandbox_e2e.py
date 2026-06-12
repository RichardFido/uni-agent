"""最小端到端测试：YR 沙箱能否正常执行 OPENYUANRONG_ENV_PREPARE_CMD 并启动 swerex"""

import asyncio
import os

# 加载环境变量（_configure_env 会自动将 OPENYUANRONG_* 映射为 AKERNEL_*）
os.environ.update({
    "DEPLOYMENT": "openyuanrong",
    "OPENYUANRONG_SERVER_ADDRESS": os.environ.get("OPENYUANRONG_SERVER_ADDRESS", ""),
    "OPENYUANRONG_ENV_PREPARE_CMD": "pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && pip config set install.trusted-host repo.huaweicloud.com && python3 -m pip install -q swe-rex",
    #"OPENYUANRONG_ENV_PREPARE_CMD": "/opt/swerex-venv/bin/python3 -m pip install -q swe-rex",

})

IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-bench-verified/sweb.eval.x86_64.astropy_1776_astropy-12907:v2"
#IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-rebench/12rambau_1776_sepal_ui-814:latest"
#IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/r2e-gym-subset/8826e2e4a3:latest"

async def main():
    from akernel_sdk import Sandbox

    print(f"=== 测试1: 创建沙箱 ===")
    print(f"镜像: {IMAGE}")
    sandbox = Sandbox(
        image=IMAGE,
        cpu=2000,
        memory=4096,
        port_forwardings=[8000],
        idle_timeout=600,
    )
    print(f"沙箱创建成功, id={sandbox.sandbox_id}")

    # ── 测试2: 基础连通性 ──
    print("\n=== 测试2: 沙箱基础命令 ===")
    r = sandbox.commands.run("echo SANDBOX_OK && which python3 && python3 --version", timeout=30)
    print(f"stdout: {r.stdout}")
    print(f"stderr: {r.stderr}")
    print(f"exit_code: {r.exit_code}")

    # ── 测试3: 检查镜像里是否已有 swe-rex ──
    print("\n=== 测试3: 检查 swe-rex 是否已安装 ===")
    r = sandbox.commands.run("python3 -c 'import swerex; print(swerex.__version__)' 2>&1", timeout=30)
    print(f"stdout: {r.stdout}")
    print(f"stderr: {r.stderr}")

    # ── 测试4: pip 网络连通性 ──
    print("\n=== 测试4: pip 网络连通性 (直接用华为云镜像安装 swe-rex) ===")
    r = sandbox.commands.run(
            "unset http_proxy && unset https_proxy && "
        "pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && "
        "pip config set install.trusted-host repo.huaweicloud.com && "
        "python3 -m pip install -q swe-rex 2>&1 && "
        "echo 'INSTALL_OK'",
        timeout=120
    )
    print(f"stdout: {r.stdout}")
    print(f"stderr: {r.stderr}")
    print(f"exit_code: {r.exit_code}")

    # ── 测试5: 后台启动 swerex server ──
    print("\n=== 测试5: 后台启动 swerex server (port 8000) ===")
    sandbox.commands.run(
        "python3 -m swerex.server --host 0.0.0.0 --port 8000 --auth-token test_token_123",
        background=True,
    )
    import time
    time.sleep(10)
    url = sandbox.get_port_url(8000)
    print(f"url: {url}")

    # ── 测试6: 检查 swerex 是否在监听 ──
    print("\n=== 测试6: 检查 swerex 是否监听 ===")
    r = sandbox.commands.run(
        "python3 -c \"import urllib.request; "
        "req=urllib.request.Request('http://127.0.0.1:8000/is_alive', headers={'X-API-Key':'test_token_123'}); "
        "print(urllib.request.urlopen(req, timeout=5).read().decode())\"",
        timeout=10
    )
    print(f"stdout: {r.stdout}")
    print(f"stderr: {r.stderr}")

    # ── 测试7: 通过 port forwarding 验证 is_alive ──
    print("\n=== 测试7: 通过 port URL 验证 is_alive ===")
    url = sandbox.get_port_url(8000, internal=False).replace("http://", "https://")
    print(f"Port URL: {url}")
    import urllib.request
    import ssl
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(f"{url}/is_alive", headers={"X-API-Key": "test_token_123"})
        resp = urllib.request.urlopen(req, timeout=10, context=ctx)
        print(f"is_alive response: {resp.read().decode()}")
    except Exception as e:
        print(f"is_alive FAILED: {type(e).__name__}: {e}")
    # ── 测试8: 检查 run_in_session 是否存在 ──
    print("\n=== 测试8: 通过 gateway 检查 openapi ===")

    url = sandbox.get_port_url(8000, internal=False).replace("http://", "https://")
    openapi_url = url + "/openapi.json"

    print(f"OpenAPI URL: {openapi_url}")

    import urllib.request
    import ssl

    try:
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(openapi_url, headers={"X-API-Key": "test_token_123"})
        resp = urllib.request.urlopen(req, timeout=10, context=ctx)
        spec = resp.read().decode()
        print("openapi fetched OK")

        import json
        data = json.loads(spec)

        paths = data.get("paths", {})
        print("\n=== checking run_in_session ===")

        if "/run_in_session" in paths:
            print("FOUND run_in_session")
        else:
            print("NOT FOUND run_in_session")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    # 清理
    print("\n=== 清理: 销毁沙箱 ===")
    sandbox.kill()
    print("沙箱已销毁")

asyncio.run(main())