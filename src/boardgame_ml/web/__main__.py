"""Launcher: ``python -m boardgame_ml.web`` (or ``boardgame-ml-web``).

Starts uvicorn on a single port serving both the API and the built
frontend, and prints the URL. Everything is local: the server binds
to 127.0.0.1 by default and training/watch happen in this process.
"""

from __future__ import annotations

import argparse
import webbrowser

from boardgame_ml.web.app import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="boardgame-ml web cockpit (local web interface)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--runs", default="runs", help="Runs directory (default: ./runs)."
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the browser on start."
    )
    args = parser.parse_args(argv)

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"boardgame-ml cockpit: {url}  (runs dir: {args.runs})")
    if args.open:
        webbrowser.open(url)
    uvicorn.run(create_app(args.runs), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
