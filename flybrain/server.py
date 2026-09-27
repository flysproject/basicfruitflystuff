"""The local HTTP server: static app, state stream, command API and atlas blobs.

Deliberately built on :mod:`http.server` from the standard library. The frontend
needs exactly three things from the backend -- the atlas binaries, a live state
stream, and a way to send commands -- and all three are a few hundred lines of
stdlib, so the whole app installs with nothing but Python. No framework, no build
step, no virtualenv.

Transport design:

* **Server-Sent Events** for the state stream, not WebSocket. The traffic is
  genuinely one-directional (60 snapshots a second down, occasional user actions
  up), SSE reconnects on its own after a network blip or a backend restart, and it
  needs no handshake or framing library. Commands go back over ordinary POSTs.
* State is published on a lock-protected handoff so the HTTP threads never touch
  live simulation objects. They read a snapshot the kernel already serialised.
* Payloads are split by rate: the hot fields ride every frame, the heavy ones
  (environment geometry, plumes, graph windows) ride at a fraction of that.
"""

from __future__ import annotations

import json
import mimetypes
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .atlas import Atlas
from .connectome import MUTATION_INFO
from .env import ENVIRONMENT_ORDER, ENVIRONMENTS, STIMULUS_CATALOG
from .fly import MODE_INFO
from .kernel import PRESET_SCENARIOS
from .neuro import MODULATORS
from .telemetry import SERIES_SCHEMA

__all__ = ["AppServer", "serve"]

STATIC_ROOT = "web"
STREAM_HZ = 60.0
SLOW_EVERY = 6  # environment geometry and plumes at ~10 Hz
GLANCE_EVERY = 12  # graphs, memory and log at ~5 Hz


class Server(ThreadingHTTPServer):
    """HTTP server that does not treat a dropped client as a fault.

    ``protocol_version = HTTP/1.1`` turns on keep-alive, so each connection thread
    parks in ``readline()`` waiting for the next request. Reloading the page or
    navigating away aborts that socket, which arrives as ``ConnectionAbortedError``
    and would otherwise print a full traceback on every reload. That is ordinary
    client behaviour, so it is suppressed; anything else still reports normally.
    """

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def find_free_port(preferred: int = 8777, host: str = "127.0.0.1") -> int:
    """Return *preferred* if it is free, else an arbitrary free port.

    Other agents and threads share this machine, so the launcher must never assume
    a port is available.
    """
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
            except OSError:
                continue
            return probe.getsockname()[1]
    raise RuntimeError("could not find a free port")


class AppServer:
    """Owns the HTTP server thread and the bootstrap manifest."""

    def __init__(self, kernel, root: str = ".", host: str = "127.0.0.1", port: int = 0) -> None:
        self.kernel = kernel
        self.root = os.path.abspath(root)
        self.static_root = os.path.join(self.root, STATIC_ROOT)
        self.host = host
        self.port = port or find_free_port()
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.started_at = time.time()
        self.request_count = 0

    # -- bootstrap --------------------------------------------------------
    def manifest(self) -> dict:
        atlas: Atlas = self.kernel.atlas
        return {
            "app": {
                "name": "FlyBrain Sandbox",
                "version": "1.0.0",
                "kernelHz": 120,
                "streamHz": STREAM_HZ,
            },
            "atlas": {
                "shape": list(atlas.shape),
                "regionShape": list(atlas.region_shape),
                "spacingUm": list(atlas.spacing_um),
                "extentUm": [round(v, 2) for v in atlas.extent_um],
                "sourceShape": list(atlas.source_shape),
                "hasMask": atlas.mask is not None,
                "synthetic": bool(atlas.meta.get("synthetic")),
                "regions": [
                    {
                        **region.to_json(),
                        "coverage": atlas.meta.get("regionCoverage", {}).get(region.key, 0),
                        "tissueOverlap": atlas.meta.get("regionTissueOverlap", {}).get(
                            region.key, {}
                        ),
                    }
                    for region in atlas.regions
                ],
                "meta": atlas.meta,
                "blobs": {
                    "template": "/api/atlas/template.bin",
                    "regionIds": "/api/atlas/region-ids.bin",
                    "mask": "/api/atlas/mask.bin" if atlas.mask else None,
                },
            },
            "modulators": [
                {
                    "key": spec.key,
                    "label": spec.label,
                    "short": spec.short,
                    "color": spec.color,
                    "blurb": spec.blurb,
                    "effect": spec.effect,
                    "default": spec.default,
                }
                for spec in MODULATORS
            ],
            "modes": MODE_INFO,
            "environments": [
                {**ENVIRONMENTS[key].to_json(), "order": index}
                for index, key in enumerate(ENVIRONMENT_ORDER)
            ],
            "stimuli": [
                {
                    "kind": spec.kind,
                    "key": spec.key,
                    "label": spec.label,
                    "color": spec.color,
                    "icon": spec.icon,
                    "blurb": spec.blurb,
                    "effect": spec.effect,
                }
                for spec in STIMULUS_CATALOG
            ],
            "presets": [
                {
                    "key": key,
                    "label": preset["label"],
                    "blurb": preset["blurb"],
                    "watch": preset["watch"],
                    "environment": preset["environment"],
                }
                for key, preset in PRESET_SCENARIOS.items()
            ],
            "mutations": [
                {"key": key, **info} for key, info in MUTATION_INFO.items()
            ],
            "series": self.kernel.telemetry.signal_names(),
            "sources": {
                "connectome": self.kernel.connectome_source.describe(),
                "atlas": "VFB NRRD" if not atlas.meta.get("synthetic") else "synthetic",
            },
        }

    # -- lifecycle --------------------------------------------------------
    def start(self) -> str:
        handler = _make_handler(self)
        self.httpd = Server((self.host, self.port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="flybrain-http",
                                       daemon=True)
        self.thread.start()
        return f"http://{self.host}:{self.port}/"

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread:
            self.thread.join(timeout=2.0)


def _make_handler(app: AppServer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FlyBrain/1.0"
        protocol_version = "HTTP/1.1"

        # Keep the console readable: one line per request is noise at 60 Hz.
        def log_message(self, fmt, *args):  # noqa: D102
            return

        # -- helpers ------------------------------------------------------
        def _json(self, payload, status: int = 200, cache: bool = False) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if not cache else "max-age=60")
            self.end_headers()
            self.wfile.write(body)

        def _bytes(self, payload: bytes, content_type: str, cache: bool = True) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "max-age=300" if cache else "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status=status)

        # -- routing ------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            app.request_count += 1
            parsed = urlparse(self.path)
            route = parsed.path
            query = parse_qs(parsed.query)

            if route == "/api/bootstrap":
                return self._json(app.manifest())
            if route == "/api/snapshot":
                return self._json(app.kernel.snapshot())
            if route == "/api/stream":
                return self._stream()
            if route == "/api/atlas/template.bin":
                return self._bytes(app.kernel.atlas.template, "application/octet-stream")
            if route == "/api/atlas/region-ids.bin":
                return self._bytes(app.kernel.atlas.region_ids, "application/octet-stream")
            if route == "/api/atlas/mask.bin":
                if app.kernel.atlas.mask is None:
                    return self._error(404, "no registered mask loaded")
                return self._bytes(app.kernel.atlas.mask, "application/octet-stream")
            if route == "/api/log":
                since = int(query.get("since", ["0"])[0])
                return self._json({"entries": app.kernel.telemetry.log.since(since, 200)})
            if route == "/api/export/csv":
                return self._download(
                    app.kernel.telemetry.export_csv(), "text/csv", "flybrain-telemetry.csv"
                )
            if route == "/api/export/json":
                return self._download(
                    app.kernel.telemetry.export_json(
                        {
                            "environment": app.kernel.world.environment.key,
                            "endocrine": app.kernel.neuro.to_json(),
                            "mutations": app.kernel.mutations.to_json(),
                            "memory": app.kernel.memory.to_json(),
                            "atlas": {"meta": app.kernel.atlas.meta},
                        }
                    ),
                    "application/json",
                    "flybrain-session.json",
                )
            if route == "/api/profile":
                return self._download(
                    json.dumps(app.kernel.neuro.targets(), indent=2),
                    "application/json",
                    "endocrine-profile.json",
                )
            return self._static(route)

        def do_POST(self) -> None:  # noqa: N802
            app.request_count += 1
            route = urlparse(self.path).path
            if route != "/api/command":
                return self._error(404, "unknown endpoint")
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 1_000_000:
                return self._error(400, "bad body length")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as error:
                return self._error(400, f"invalid JSON: {error}")
            name = payload.get("name")
            if not name:
                return self._error(400, "missing command name")
            app.kernel.post(str(name), payload.get("payload") or {})
            return self._json({"ok": True, "queued": name})

        def _download(self, text: str, content_type: str, filename: str) -> None:
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, route: str) -> None:
            relative = "index.html" if route in ("/", "") else route.lstrip("/")
            # Contain the path: a served file must live under web/.
            target = os.path.normpath(os.path.join(app.static_root, relative))
            if not target.startswith(app.static_root):
                return self._error(403, "forbidden")
            if os.path.isdir(target):
                target = os.path.join(target, "index.html")
            if not os.path.isfile(target):
                return self._error(404, f"not found: {relative}")
            content_type = mimetypes.guess_type(target)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in (
                "application/javascript",
                "application/json",
            ):
                content_type += "; charset=utf-8"
            with open(target, "rb") as handle:
                return self._bytes(handle.read(), content_type, cache=False)

        # -- SSE ----------------------------------------------------------
        def _stream(self) -> None:
            """Stream snapshots as Server-Sent Events.

            Each connection keeps its own log cursor, so a reconnecting client gets
            the events it missed without disturbing anyone else.
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            kernel = app.kernel
            cursor = 0
            frame = 0
            interval = 1.0 / STREAM_HZ
            try:
                while True:
                    frame_started = time.perf_counter()
                    fast = kernel.last_snapshot
                    if not fast:
                        time.sleep(interval)
                        continue
                    payload = dict(fast)
                    if frame % SLOW_EVERY == 0:
                        payload.update(kernel.last_slow)
                    if frame % GLANCE_EVERY == 0:
                        payload["glance"] = {
                            "graphs": kernel.telemetry.graphs(30.0),
                            "memory": kernel.memory.to_json(),
                            "stats": dict(kernel.telemetry.stats),
                        }
                    entries = kernel.telemetry.log.since(cursor, 40)
                    if entries:
                        cursor = entries[-1]["seq"]
                        payload["events"] = entries

                    chunk = "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
                    frame += 1
                    elapsed = time.perf_counter() - frame_started
                    if elapsed < interval:
                        time.sleep(interval - elapsed)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                # Client navigated away or the browser tab closed.
                return

    return Handler


def serve(kernel, root: str = ".", host: str = "127.0.0.1", port: int = 0) -> AppServer:
    app = AppServer(kernel, root=root, host=host, port=port)
    return app
