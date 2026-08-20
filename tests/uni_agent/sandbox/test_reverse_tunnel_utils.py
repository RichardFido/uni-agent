"""Tests for the sandbox reverse-tunnel URL helpers.

These cover the pure string math shared by sandbox-facing callers: rewriting a
gateway URL to the in-sandbox tunnel and extracting the tunnel ``upstream``.
"""

from __future__ import annotations

from uni_agent.sandbox.reverse_tunnel_utils import DEFAULT_PROXY_PORT, extract_upstream, rewrite_gateway_url


def test_rewrite_gateway_url_replaces_host_with_tunnel_port():
    assert rewrite_gateway_url("http://gateway.example:40169/sessions/abc/v1") == (
        f"http://127.0.0.1:{DEFAULT_PROXY_PORT}/sessions/abc/v1"
    )


def test_rewrite_gateway_url_custom_proxy_port():
    assert rewrite_gateway_url("http://gateway:8000/v1", proxy_port=4242) == "http://127.0.0.1:4242/v1"


def test_rewrite_gateway_url_strip_v1_drops_trailing_path():
    assert rewrite_gateway_url("http://gateway:8000/v1", strip_v1=True) == "http://127.0.0.1:38197"


def test_extract_upstream_returns_host_port():
    assert extract_upstream("http://gateway.example:40169/sessions/abc/v1") == "gateway.example:40169"


def test_extract_upstream_none_without_port():
    assert extract_upstream("http://gateway/v1") is None
