"""Start the private THREADORA studio:  python run_web.py  [--port 8765] [--no-browser]

Listens on 127.0.0.1 only (this computer). To use it from another device on your own network,
set THREADORA_ACCESS_TOKEN first and pass --host 0.0.0.0.
"""
import argparse
import os
import sys
import threading
import webbrowser

import uvicorn


def main():
    os.environ.setdefault("PYTHONUTF8", "1")   # UTF-8 files/logs on Windows for the engine's test runs
    ap = argparse.ArgumentParser(description="THREADORA private web studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost") and not os.environ.get("THREADORA_ACCESS_TOKEN"):
        sys.exit("Refusing to listen beyond this computer without THREADORA_ACCESS_TOKEN set.")
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}/"
    print(f"THREADORA studio: {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("webapp.server:create_app", factory=True, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
