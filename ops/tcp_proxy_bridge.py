#!/usr/bin/env python3
"""Forward a TCP listener without dropping nonblocking writes under load."""

from __future__ import annotations

import argparse
import select
import socket
import socketserver


IO_TIMEOUT_SECONDS = 60.0
BUFFER_SIZE = 1024 * 1024


def send_all(destination: socket.socket, data: bytes) -> None:
    pending = memoryview(data)
    while pending:
        _, writable, errored = select.select((), (destination,), (destination,), IO_TIMEOUT_SECONDS)
        if errored or not writable:
            raise TimeoutError("TCP proxy destination was not writable")
        try:
            sent = destination.send(pending)
        except BlockingIOError:
            continue
        if sent <= 0:
            raise ConnectionError("TCP proxy destination closed during write")
        pending = pending[sent:]


class ForwardingHandler(socketserver.BaseRequestHandler):
    target: tuple[str, int]

    def handle(self) -> None:
        try:
            with socket.create_connection(self.target, timeout=15) as upstream:
                peers = (self.request, upstream)
                for peer in peers:
                    peer.setblocking(False)
                while True:
                    readable, _, errored = select.select(peers, (), peers, IO_TIMEOUT_SECONDS)
                    if errored:
                        return
                    if not readable:
                        continue
                    for source in readable:
                        try:
                            data = source.recv(BUFFER_SIZE)
                        except BlockingIOError:
                            continue
                        if not data:
                            return
                        destination = upstream if source is self.request else self.request
                        send_all(destination, data)
        except (ConnectionError, OSError, TimeoutError):
            return


class ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, required=True)
    args = parser.parse_args()

    ForwardingHandler.target = (args.target_host, args.target_port)
    with ThreadingServer((args.listen_host, args.listen_port), ForwardingHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
