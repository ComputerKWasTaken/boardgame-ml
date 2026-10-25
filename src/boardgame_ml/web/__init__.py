"""The boardgame-ml web cockpit: a local web interface to the framework.

Backend (FastAPI) wrapping the framework's own contracts — training
jobs run the framework's trainer, watch sessions step envs with the
evaluator's greedy ``predict_action`` — plus the built React frontend,
served by this same process. Runs/checkpoints stay on disk in
``runs/``; the browser's localStorage holds only the personal layer
(preferences, starred runs, saved replays, history).

Launch with ``python -m boardgame_ml.web`` (see ``__main__.py``).
"""

from __future__ import annotations

try:
    from boardgame_ml.web.app import create_app
except ModuleNotFoundError as exc:  # fastapi/uvicorn not installed
    raise ModuleNotFoundError(
        f"{exc}. The web cockpit needs its extra dependencies: "
        'pip install -e ".[web]"'
    ) from exc

__all__ = ["create_app"]
