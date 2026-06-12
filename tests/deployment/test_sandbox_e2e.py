"""Minimal e2e test: Can YR sandbox execute OPENYUANRONG_ENV_PREPARE_CMD and launch swerex"""

import asyncio
import os

# Load environment variables (_configure_env automatically maps OPENYUANRONG_* to AKERNEL_*)
os.environ.update({
    "AKERNEL_SERVER_ADDRESS": os.environ.get("OPENYUANRONG_SERVER_ADDRESS", ""),
    "AKERNEL_TOKEN": os.environ.get("OPENYUANRONG_TOKEN", ""),

})

IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-bench-verified/sweb.eval.x86_64.astropy_1776_astropy-12907:v2"
#IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-rebench/12rambau_1776_sepal_ui-814:latest"
#IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/r2e-gym-subset/8826e2e4a3:latest"

async def main():
    from akernel_sdk import Sandbox

    print(f"=== Test 1: Create sandbox ===")
    print(f"Image: {IMAGE}")
    sandbox = Sandbox(
        image=IMAGE,
        cpu=2000,
        memory=4096,
        port_forwardings=[8000],
        idle_timeout=600,
    )
    print(f"Sandbox created successfully, id={sandbox.sandbox_id}")

    # ── Test 2: Basic connectivity ──
    print("\n=== Test 2: Sandbox basic commands ===")
    r = sandbox.commands.run("echo SANDBOX_OK && which python3 && python3 --version", timeout=30)
    print(f"stdout: {r.stdout}")
    print(f"stderr: {r.stderr}")
    print(f"exit_code: {r.exit_code}")

    # ── Test 3: Check if swe-rex is already installed in the image ──
    print("\n=== Test 3: Check if swe-rex is already installed ===")
    r = sandbox.commands.run("python3 -c 'import swerex; print(swerex.__version__)' 2>&1", timeout=30)
    print(f"stdout: {r.stdout}")
    print(f"stderr: {r.stderr}")

    # ── Test 4: pip network connectivity ──
    print("\n=== Test 4: pip network connectivity (install swe-rex via Huawei cloud mirror) ===")
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

    # ── Test 5: Launch swerex server in background ──
    print("\n=== Test 5: Launch swerex server in background (port 8000) ===")
    sandbox.commands.run(
        "python3 -m swerex.server --host 0.0.0.0 --port 8000 --auth-token test_token_123",
        background=True,
    )
    import time
    time.sleep(10)
    url = sandbox.get_port_url(8000)
    print(f"url: {url}")

    # ── Test 6: Check if swerex is listening ──
    print("\n=== Test 6: Check if swerex is listening ===")
    r = sandbox.commands.run(
        "python3 -c \"import urllib.request; "
        "req=urllib.request.Request('http://127.0.0.1:8000/is_alive', headers={'X-API-Key':'test_token_123'}); "
        "print(urllib.request.urlopen(req, timeout=5).read().decode())\"",
        timeout=10
    )
    print(f"stdout: {r.stdout}")
    print(f"stderr: {r.stderr}")

    # ── Test 7: Verify is_alive via port forwarding ──
    print("\n=== Test 7: Verify is_alive via port URL ===")
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
    # ── Test 8: Check if run_in_session exists ──
    print("\n=== Test 8: Check openapi via gateway ===")

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

    # Cleanup
    print("\n=== Cleanup: Destroy sandbox ===")
    sandbox.kill()
    print("Sandbox destroyed")

asyncio.run(main())
