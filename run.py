#!/usr/bin/env python3
"""FlyBrain Sandbox launcher.

One command, no dependency install:

    python run.py

What happens, in order:

1. The VFB atlas is loaded from cache or baked from the NRRD volumes in the
   workspace (about five seconds cold, instant afterwards).
2. The simulation kernel starts on its own 120 Hz thread.
3. A local HTTP server starts on the first free port.
4. A window opens around the app.

The window tries, in order: an installed ``pywebview`` (a genuine chromeless
desktop window using the OS webview), then an installed PySide6 ``QWebEngineView``,
then the default browser. All three are optional -- the app runs identically in
any of them, and having zero required packages is the point.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flybrain.atlas import load_or_build  # noqa: E402
from flybrain.kernel import Kernel  # noqa: E402
from flybrain.server import AppServer, find_free_port  # noqa: E402

BANNER = r"""
  ______ _       ____            _         _____                 _ _
 |  ____| |     |  _ \          (_)       / ____|               | | |
 | |__  | |_   _| |_) |_ __ __ _ _ _ __  | (___   __ _ _ __   __| | |__   ___  _ __
 |  __| | | | | |  _ <| '__/ _` | | '_ \  \___ \ / _` | '_ \ / _` | '_ \ / _ \| '_ \
 | |    | | |_| | |_) | | | (_| | | | | | ____) | (_| | | | | (_| | |_) | (_) | | | |
 |_|    |_|\__, |____/|_|  \__,_|_|_| |_||_____/ \__,_|_| |_|\__,_|_.__/ \___/|_| |_|
            __/ |
           |___/    Drosophila neural sandbox
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FlyBrain Sandbox desktop app.")
    parser.add_argument("--root", default=ROOT, help="workspace root holding the VFB volumes")
    parser.add_argument("--environment", default="kitchen", help="starting environment key")
    parser.add_argument("--flies", type=int, default=1, help="number of flies (1-10)")
    parser.add_argument("--port", type=int, default=8777, help="preferred port, 0 for any")
    parser.add_argument("--connectome", default=None, help="path to a region-circuit JSON file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-window", action="store_true",
                        help="serve only; do not open a window")
    parser.add_argument("--headless", action="store_true",
                        help="run the kernel only, print a text report, and exit")
    parser.add_argument("--headless-seconds", type=float, default=60.0,
                        help="simulated seconds for --headless")
    return parser.parse_args()


def run_native_window(url: str, title: str) -> str | None:
    """Try to open a real desktop window. Returns the shell name, or None.

    Both options must run on the **main** thread, so this blocks until the user
    closes the window. Anything that fails here is not an error: the app is fully
    usable in a browser tab, and zero required packages is a deliberate property.
    """
    try:
        import webview  # type: ignore

        webview.create_window(title, url, width=1620, height=1000, min_size=(1100, 700))
        webview.start()
        return "pywebview"
    except ImportError:
        pass
    except Exception as error:  # pragma: no cover - depends on the host desktop
        print(f"  ! pywebview window failed ({error}); trying the browser")

    try:
        from PySide6.QtCore import QUrl  # type: ignore
        from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore
        from PySide6.QtWidgets import QApplication  # type: ignore

        application = QApplication.instance() or QApplication(sys.argv)
        view = QWebEngineView()
        view.setWindowTitle(title)
        view.resize(1620, 1000)
        view.load(QUrl(url))
        view.show()
        application.exec()
        return "PySide6 QWebEngineView"
    except ImportError:
        pass
    except Exception as error:  # pragma: no cover
        print(f"  ! PySide6 window failed ({error}); trying the browser")
    return None


def open_in_browser(url: str) -> None:
    import webbrowser

    webbrowser.open(url)
    print("  ... opened in your default browser")
    print("  ... (install pywebview or PySide6 for a chromeless desktop window)")


def run_headless(kernel: Kernel, seconds: float) -> int:
    """Run the kernel with no UI and print a behavioural report.

    This is the same simulate loop the app uses, minus the transport, which makes it
    useful for smoke tests and for batch experiments, and it is what a CI check
    should run.
    """
    import collections

    step = 1.0 / 120.0
    modes: collections.Counter = collections.Counter()
    region_peak: dict[str, float] = {key: 0.0 for key in kernel.atlas_regions}
    started = time.perf_counter()
    steps = int(seconds * 120)
    for index in range(steps):
        kernel._step(step)
        if index % 40 == 0:
            modes[kernel.primary.mode] += 1
            for key in kernel.atlas_regions:
                region_peak[key] = max(region_peak[key], kernel.router.signal(key))
    elapsed = time.perf_counter() - started
    primary = kernel.primary

    print(f"\n  simulated      {seconds:.0f} s in {elapsed:.2f} s wall "
          f"({100.0 * elapsed / max(1e-9, seconds):.1f}% of the realtime budget)")
    print(f"  steps          {steps} at 120 Hz")
    print(f"  mode mix       " + ", ".join(
        f"{key} {100.0 * value / max(1, sum(modes.values())):.0f}%" for key, value in modes.most_common()
    ))
    print(f"  distance       {primary.distance_travelled:.0f} mm")
    print(f"  escapes        {primary.escape_events}")
    print(f"  feeds          {kernel.telemetry.stats['feeds']}  "
          f"grooming bouts {kernel.telemetry.stats['groomingEvents']}")
    print(f"  physiology     energy {kernel.neuro.state.energy:.2f}  "
          f"hunger {kernel.neuro.state.hunger:.2f}  fear {kernel.neuro.state.fear:.2f}")
    print(f"  region peaks   " + ", ".join(
        f"{key} {value:.2f}" for key, value in sorted(region_peak.items(), key=lambda kv: -kv[1])
    ))
    print("\n  recent event log:")
    for entry in kernel.telemetry.log.tail(10):
        print(f"    [{entry['t']:7.2f}] {entry['level']:<6} {entry['message'][:110]}")
    return 0


def main() -> int:
    args = parse_args()
    print(BANNER)

    atlas = load_or_build(
        args.root,
        progress=lambda message: print(f"  ... {message}"),
    )
    if atlas.meta.get("synthetic"):
        print("  ! no VFB volumes found: running on a synthetic CNS stand-in")
    else:
        orientation = atlas.meta.get("orientation", {})
        print(
            f"  ... atlas {atlas.shape[0]}x{atlas.shape[1]}x{atlas.shape[2]} from "
            f"{atlas.source_shape[0]}x{atlas.source_shape[1]}x{atlas.source_shape[2]} source, "
            f"midline axis '{orientation.get('midlineAxis', '?')}', "
            f"{len(atlas.regions)} neuropils"
        )

    kernel = Kernel(
        atlas,
        root=args.root,
        environment=args.environment,
        fly_count=args.flies,
        connectome_path=args.connectome,
    )

    if args.headless:
        code = run_headless(kernel, args.headless_seconds)
        kernel.shutdown()
        return code

    app = AppServer(kernel, root=args.root, host=args.host,
                    port=args.port or find_free_port())
    url = app.start()
    kernel.start()

    print(f"\n  kernel  120 Hz on a dedicated thread")
    print(f"  stream  {url}api/stream")
    print(f"  app     {url}")
    print("\n  Press Ctrl+C to stop.\n")

    if args.no_window:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    else:
        shell = run_native_window(url, "FlyBrain Sandbox")
        if shell:
            print(f"  window shell {shell} closed")
        else:
            open_in_browser(url)
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass

    kernel.shutdown()
    app.stop()
    print("  stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
