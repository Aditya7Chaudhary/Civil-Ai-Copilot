import argparse
import contextlib
import os
import socket
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def _pick_port(host: str, preferred: int, attempts: int) -> int:
    for port in range(preferred, preferred + max(1, attempts)):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the static frontend with a local URL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument(
        "--port-attempts",
        type=int,
        default=25,
        help="How many consecutive ports to try starting from --port",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the frontend URL in your default browser",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frontend_dir = os.path.join(base_dir, "app", "frontend")
    index_path = os.path.join(frontend_dir, "index.html")

    if not os.path.isdir(frontend_dir) or not os.path.isfile(index_path):
        print(f"Frontend not found at: {frontend_dir}", file=sys.stderr)
        return 1

    port = _pick_port(args.host, args.port, args.port_attempts)
    handler = lambda *h_args, **h_kwargs: SimpleHTTPRequestHandler(  # noqa: E731
        *h_args, directory=frontend_dir, **h_kwargs
    )
    httpd = ThreadingHTTPServer((args.host, port), handler)

    url = f"http://{args.host}:{httpd.server_address[1]}/index.html"
    print("Frontend running at:")
    print(url)
    print("Press Ctrl+C to stop.")

    if args.open:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()


if __name__ == "__main__":
    raise SystemExit(main())

