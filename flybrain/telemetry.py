"""Telemetry: ring-buffered time series, the event log, and export.

The kernel runs at 120 Hz but nothing is worth recording at that rate. Series are
decimated to a fixed 20 Hz on write, which keeps a 90-second window at ~3.6 kB per
channel while still showing the shape of a transect. Events are kept verbatim
because the log is the app's narrative, not its data.

Export is deliberately plain: CSV for the series, JSON for everything else, so the
output drops straight into pandas or R without a bespoke reader.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field

__all__ = ["Series", "EventLog", "Telemetry", "SERIES_SCHEMA"]

# key -> (label, unit, colour, default-visible)
SERIES_SCHEMA: dict[str, tuple[str, str, str, bool]] = {
    "speed": ("Velocity", "mm/s", "#38bdf8", True),
    "hunger": ("Hunger", "0-1", "#a3e635", True),
    "fear": ("Fear index", "0-1", "#f87171", True),
    "odor": ("Odor concentration", "a.u.", "#fbbf24", True),
    "energy": ("Energy", "0-1", "#c084fc", False),
    "arousal": ("Arousal", "0-1", "#ff7a45", False),
    "DA": ("Dopamine", "0-1", "#f9d423", False),
    "OA": ("Octopamine", "0-1", "#ff7a45", False),
    "5HT": ("Serotonin", "0-1", "#7dd3fc", False),
    "NPF": ("Neuropeptide F", "0-1", "#a3e635", False),
    "DILP": ("DILP", "0-1", "#c084fc", False),
    "CLOCK": ("Circadian", "0-1", "#5eead4", False),
    "GF": ("Giant fibre", "0-1", "#f59e0b", False),
    "CX": ("Central complex", "0-1", "#fb7185", False),
    "AL": ("Antennal lobe", "0-1", "#4ade80", False),
    "MB": ("Mushroom body", "0-1", "#c084fc", False),
}

# How many samples at 20 Hz: 90 seconds of history.
SERIES_CAPACITY = 1800
SAMPLE_HZ = 20.0


@dataclass
class Series:
    key: str
    capacity: int = SERIES_CAPACITY
    times: deque = field(default_factory=lambda: deque(maxlen=SERIES_CAPACITY))
    values: deque = field(default_factory=lambda: deque(maxlen=SERIES_CAPACITY))

    def push(self, t: float, value: float) -> None:
        self.times.append(round(t, 3))
        self.values.append(round(value, 5))

    def latest(self) -> float:
        return self.values[-1] if self.values else 0.0

    def window(self, seconds: float, now: float) -> tuple[list[float], list[float]]:
        if not self.times:
            return ([], [])
        cutoff = now - seconds
        xs: list[float] = []
        ys: list[float] = []
        # Walk backwards so a long window does not copy the whole ring.
        for index in range(len(self.times) - 1, -1, -1):
            if self.times[index] < cutoff:
                break
            xs.append(self.times[index])
            ys.append(self.values[index])
        xs.reverse()
        ys.reverse()
        return xs, ys


class EventLog:
    """A bounded, natural-language log of what the simulation actually did."""

    LEVELS = ("info", "event", "alert", "learn")

    def __init__(self, capacity: int = 400) -> None:
        self.capacity = capacity
        self.entries: deque = deque(maxlen=capacity)
        self.sequence = 0

    def add(self, t: float, message: str, level: str = "info", channel: str = "brain",
            fly: str | None = None, extra: dict | None = None) -> dict:
        self.sequence += 1
        entry = {
            "seq": self.sequence,
            "t": round(t, 2),
            "level": level if level in self.LEVELS else "info",
            "channel": channel,
            "fly": fly,
            "message": message,
        }
        if extra:
            entry["extra"] = extra
        self.entries.append(entry)
        return entry

    def since(self, sequence: int, limit: int = 60) -> list[dict]:
        return [e for e in self.entries if e["seq"] > sequence][-limit:]

    def tail(self, count: int = 40) -> list[dict]:
        return list(self.entries)[-count:]

    def to_json(self) -> dict:
        return {"entries": list(self.entries), "sequence": self.sequence}


class Telemetry:
    """All recorded history, plus the export surface."""

    def __init__(self, capacity: int = SERIES_CAPACITY) -> None:
        self.capacity = capacity
        self.series: dict[str, Series] = {
            key: Series(key=key, capacity=capacity) for key in SERIES_SCHEMA
        }
        self.log = EventLog()
        self.sample_accumulator = 0.0
        self.now = 0.0
        self.events_total = 0
        # Aggregate counters surfaced as "session stats" in the UI.
        self.stats = {
            "escapes": 0,
            "feeds": 0,
            "groomingEvents": 0,
            "conditioningTrials": 0,
            "distanceMm": 0.0,
            "sleepSeconds": 0.0,
            "flightSeconds": 0.0,
        }

    # -- recording --------------------------------------------------------
    def record(self, t: float, values: dict[str, float], dt: float) -> None:
        """Decimate *values* onto the fixed sample rate."""
        self.now = t
        self.sample_accumulator += dt
        target = 1.0 / SAMPLE_HZ
        if self.sample_accumulator < target:
            return
        self.sample_accumulator = 0.0
        for key, series in self.series.items():
            if key in values:
                series.push(t, float(values[key]))

    def headroom(self, key: str) -> tuple[float, float]:
        series = self.series.get(key)
        if not series or not series.values:
            return (0.0, 1.0)
        low = min(series.values)
        high = max(series.values)
        if high - low < 1e-6:
            high = low + 1.0
        return (low, high)

    # -- serialisation ----------------------------------------------------
    def graphs(self, seconds: float = 30.0, keys: list[str] | None = None) -> dict:
        """Everything the telemetry panel needs to draw, ready to plot."""
        out: dict[str, dict] = {}
        for key in keys or [k for k, v in SERIES_SCHEMA.items() if v[3]]:
            series = self.series.get(key)
            if series is None:
                continue
            xs, ys = series.window(seconds, self.now)
            label, unit, color, _visible = SERIES_SCHEMA[key]
            out[key] = {
                "label": label,
                "unit": unit,
                "color": color,
                "t": xs,
                "v": ys,
                "latest": round(ys[-1], 4) if ys else 0.0,
                "range": [round(min(ys), 4), round(max(ys), 4)] if ys else [0.0, 1.0],
            }
        return out

    def signal_names(self) -> list[dict]:
        return [
            {"key": key, "label": label, "unit": unit, "color": color, "default": visible}
            for key, (label, unit, color, visible) in SERIES_SCHEMA.items()
        ]

    # -- export -----------------------------------------------------------
    def export_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        keys = list(self.series.keys())
        writer.writerow(["time_s"] + [f"{k}_{SERIES_SCHEMA[k][1]}" for k in keys])
        lengths = [len(self.series[k].times) for k in keys]
        rows = max(lengths) if lengths else 0
        # Series are written on a shared clock but can be shorter than one another,
        # so index from the end of each ring and pad rather than zipping.
        for row in range(rows):
            line = []
            time_value = ""
            for key in keys:
                times = self.series[key].times
                values = self.series[key].values
                offset = row - (rows - len(times))
                if 0 <= offset < len(times):
                    if time_value == "":
                        time_value = f"{times[offset]:.3f}"
                    line.append(f"{values[offset]:.5f}")
                else:
                    line.append("")
            writer.writerow([time_value] + line)
        return buffer.getvalue()

    def export_json(self, extra: dict | None = None) -> str:
        payload = {
            "exportedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "durationS": round(self.now, 2),
            "schema": {
                key: {"label": label, "unit": unit}
                for key, (label, unit, _color, _visible) in SERIES_SCHEMA.items()
            },
            "stats": self.stats,
            "log": list(self.log.entries),
        }
        if extra:
            payload.update(extra)
        return json.dumps(payload, indent=2)

    def export_series_json(self) -> str:
        return json.dumps(
            {
                key: {"t": list(series.times), "v": list(series.values)}
                for key, series in self.series.items()
            }
        )

    def write_exports(self, directory: str, stem: str = "flybrain-session",
                      extra: dict | None = None) -> list[str]:
        os.makedirs(directory, exist_ok=True)
        written = []
        for suffix, content in (
            ("csv", self.export_csv()),
            ("json", self.export_json(extra)),
        ):
            path = os.path.join(directory, f"{stem}.{suffix}")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            written.append(path)
        return written

    def reset(self) -> None:
        for series in self.series.values():
            series.times.clear()
            series.values.clear()
        self.log.entries.clear()
        self.sample_accumulator = 0.0
        self.now = 0.0
        for key in self.stats:
            self.stats[key] = 0.0 if isinstance(self.stats[key], float) else 0
