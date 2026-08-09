#!/usr/bin/env python3
from __future__ import annotations

import argparse
import select
import socket
import socketserver
import sys
import threading
import time
from urllib.parse import urlsplit


class ConnectProxyHandler(socketserver.BaseRequestHandler):
    timeout_s = 30.0
    buffer_size = 64 * 1024
    listen_port = 31080
    error_log_interval_s = 60.0
    _error_log_lock = threading.Lock()
    _last_error_log = 0.0
    _suppressed_errors = 0

    def handle(self) -> None:
        client = self.request
        client.settimeout(self.timeout_s)
        try:
            header = self._read_header(client)
            if not header:
                return
            first = header.split(b"\r\n", 1)[0].decode("latin1", errors="replace")
            parts = first.split()
            if len(parts) < 3:
                self._send_error(client, 400, "bad request")
                return
            if parts[0].upper() != "CONNECT":
                self._handle_http_request(client, header, parts)
                return
            host, port = self._parse_authority(parts[1])
            if self._is_self_target(host, port):
                self._send_error(client, 508, "proxy loop detected")
                return
            with self._create_ipv4_connection(host, port) as upstream:
                upstream.settimeout(self.timeout_s)
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                self._tunnel(client, upstream)
        except Exception as exc:  # noqa: BLE001 - keep proxy alive after bad client connections.
            self._log_error(exc)

    @classmethod
    def _is_self_target(cls, host: str, port: int) -> bool:
        return port == cls.listen_port and host.lower() in {"127.0.0.1", "localhost", "::1"}

    @classmethod
    def _log_error(cls, exc: Exception) -> None:
        now = time.monotonic()
        with cls._error_log_lock:
            cls._suppressed_errors += 1
            if now - cls._last_error_log < cls.error_log_interval_s:
                return
            suppressed = cls._suppressed_errors - 1
            cls._suppressed_errors = 0
            cls._last_error_log = now
        suffix = f"; suppressed={suppressed}" if suppressed else ""
        print(
            f"connect_proxy error: {type(exc).__name__}: {exc}{suffix}",
            file=sys.stderr,
            flush=True,
        )

    def _read_header(self, client: socket.socket) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while total < self.buffer_size:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            data = b"".join(chunks)
            if b"\r\n\r\n" in data:
                return data
        return b"".join(chunks)

    @staticmethod
    def _parse_authority(value: str) -> tuple[str, int]:
        if value.startswith("["):
            host, _, rest = value[1:].partition("]")
            if not rest.startswith(":"):
                raise ValueError(f"missing port in CONNECT target: {value!r}")
            return host, int(rest[1:])
        host, sep, raw_port = value.rpartition(":")
        if not sep or not host:
            raise ValueError(f"missing port in CONNECT target: {value!r}")
        return host, int(raw_port)

    @staticmethod
    def _send_error(client: socket.socket, status: int, reason: str) -> None:
        body = f"{status} {reason}\n".encode("utf-8")
        client.sendall(
            f"HTTP/1.1 {status} {reason}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode(
                "ascii"
            )
            + body
        )

    def _handle_http_request(self, client: socket.socket, request: bytes, parts: list[str]) -> None:
        head, sep, body = request.partition(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        method, target, version = parts[0], parts[1], parts[2]
        parsed = urlsplit(target)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if parsed.scheme and parsed.scheme.lower() != "http":
            self._send_error(client, 400, "unsupported proxy target")
            return
        if not host:
            for line in lines[1:]:
                name, _, value = line.partition(b":")
                if name.lower() == b"host":
                    host_value = value.strip().decode("latin1", errors="replace")
                    host, raw_port = self._parse_host_header(host_value)
                    port = raw_port or 80
                    break
        if not host:
            self._send_error(client, 400, "missing host")
            return
        if self._is_self_target(host, port):
            self._send_error(client, 508, "proxy loop detected")
            return
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        rewritten = [f"{method} {path} {version}".encode("latin1")]
        for line in lines[1:]:
            lower = line.split(b":", 1)[0].strip().lower()
            if lower in {b"proxy-connection", b"connection"}:
                continue
            rewritten.append(line)
        rewritten.append(b"Connection: close")
        outbound = b"\r\n".join(rewritten) + b"\r\n\r\n" + body
        with self._create_ipv4_connection(host, port) as upstream:
            upstream.settimeout(self.timeout_s)
            upstream.sendall(outbound)
            self._pipe_response(client, upstream)

    def _create_ipv4_connection(self, host: str, port: int) -> socket.socket:
        last_error: OSError | None = None
        for family, socktype, proto, _, sockaddr in socket.getaddrinfo(
            host, port, family=socket.AF_INET, type=socket.SOCK_STREAM
        ):
            sock = socket.socket(family, socktype, proto)
            sock.settimeout(self.timeout_s)
            try:
                sock.connect(sockaddr)
                return sock
            except OSError as exc:
                last_error = exc
                sock.close()
        if last_error is not None:
            raise last_error
        raise OSError(f"no IPv4 address for {host}:{port}")

    @staticmethod
    def _parse_host_header(value: str) -> tuple[str, int | None]:
        if value.startswith("["):
            host, _, rest = value[1:].partition("]")
            if rest.startswith(":"):
                return host, int(rest[1:])
            return host, None
        host, sep, raw_port = value.rpartition(":")
        if sep and raw_port.isdigit():
            return host, int(raw_port)
        return value, None

    def _pipe_response(self, client: socket.socket, upstream: socket.socket) -> None:
        while True:
            data = upstream.recv(self.buffer_size)
            if not data:
                return
            client.sendall(data)

    def _tunnel(self, client: socket.socket, upstream: socket.socket) -> None:
        sockets = [client, upstream]
        while True:
            readable, _, errored = select.select(sockets, (), sockets, self.timeout_s)
            if errored or not readable:
                return
            for sock in readable:
                data = sock.recv(self.buffer_size)
                if not data:
                    return
                target = upstream if sock is client else client
                target.sendall(data)


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 256


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal CONNECT proxy for Docker daemon pulls")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31080)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    ConnectProxyHandler.timeout_s = max(1.0, float(args.timeout_s))
    ConnectProxyHandler.listen_port = args.port
    with ThreadingTCPServer((args.host, args.port), ConnectProxyHandler) as server:
        print(f"connect_proxy listening on {args.host}:{args.port}", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
