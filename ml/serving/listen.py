"""Run a FastAPI app under uvicorn, dual-stack when asked to listen on ``::``.

Railway's private network is IPv6 and its health checks and public proxy
reach the container over IPv4, so the servers must accept both. uvicorn
binds a plain IPv6 socket for ``::`` and leaves IPV6_V6ONLY to the kernel
default; this binds the socket itself with dual-stack on and hands it over.
"""

from __future__ import annotations

import socket


def serve(app, host: str, port: int, log_level: str = "warning") -> None:
    import uvicorn

    if host in ("::", "*", "dual"):
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.bind(("::", port))
        sock.listen(2048)
        print(f"listening on [::]:{port} (dual-stack)", flush=True)
        uvicorn.run(app, fd=sock.fileno(), log_level=log_level)
        return
    print(f"listening on {host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
