"""Reverse-tunnel URL helpers shared by sandbox-facing callers.

The openyuanrong sandbox exposes a fixed in-sandbox tunnel port (``proxy_port``,
see ``SandboxConfig.sandbox_kwargs``) through which the sandboxed process reaches
the gateway, while the gateway's real address lives on the host. These helpers
translate between the two views of the same endpoint:

- :func:`rewrite_gateway_url` -- rewrite the gateway URL to the sandbox-internal
  tunnel (``host:port`` → ``127.0.0.1:<proxy_port>``), keeping the path.
- :func:`extract_upstream` -- extract ``host:port`` to feed the tunnel's
  ``upstream`` so the provider knows where to forward the tunnel.

Both are pure string math (stdlib only) so any caller -- the framework's
``run_task`` bridge (and self-contained recipe runners) -- can share one
implementation instead of re-deriving it.
"""

from __future__ import annotations

from urllib.parse import urlparse

#: Sandbox-internal reverse-tunnel port (must match the sandbox's ``proxy_port``).
DEFAULT_PROXY_PORT = 38197


def rewrite_gateway_url(
    gateway_url: str,
    proxy_port: int = DEFAULT_PROXY_PORT,
    *,
    strip_v1: bool = False,
) -> str:
    """Rewrite a gateway URL to the sandbox-internal tunnel (``127.0.0.1:<proxy_port>``).

    Replaces host:port with ``127.0.0.1:<proxy_port>`` and keeps the path, so an
    in-sandbox endpoint reaches the gateway through the reverse tunnel. Example:
    ``http://gateway.example:40169/sessions/abc/v1`` ->
    ``http://127.0.0.1:38197/sessions/abc/v1``.
    """
    parsed = urlparse(gateway_url)
    path = parsed.path.removesuffix("/v1") if strip_v1 else parsed.path
    # The ``127.0.0.1`` loopback + fixed in-sandbox ``proxy_port`` is the
    # openyuanrong reverse-tunnel convention (the provider forwards the gateway to
    # that loopback; ``run_task`` injects the ``upstream``). A provider exposing a
    # tunnel layout other than a loopback on ``proxy_port`` would generalize this.
    return f"http://127.0.0.1:{proxy_port}{path}"


def extract_upstream(gateway_url: str) -> str | None:
    """Extract ``host:port`` from a gateway URL (the tunnel's ``upstream``).

    Returns ``None`` when the URL carries no host or port, so callers can fail
    loudly instead of forwarding a ``None:None`` upstream.
    """
    parsed = urlparse(gateway_url)
    if not parsed.hostname or not parsed.port:
        return None
    return f"{parsed.hostname}:{parsed.port}"