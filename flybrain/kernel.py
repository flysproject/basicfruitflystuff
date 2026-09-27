"""The simulation kernel: a fixed 120 Hz loop on its own thread.

The UI thread and the physics thread never touch each other's state. Commands go
one way through a lock-protected queue; state comes back as immutable snapshots.
That is what keeps the interface responsive when the fly is mid-escape and the
modulator system is firing transients every tick.

Timing model:

* **Physics** runs at a fixed :data:`STEP_HZ` on an accumulator, so the simulation
  is deterministic in step count and immune to a slow frame or a stalled HTTP
  response. A frame that overruns is caught up rather than skipped, up to a cap so
  a suspended laptop does not turn into a 10-second freeze.
* **Snapshots** are published at :data:`PUBLISH_HZ`, decoupled from the physics
  rate. The renderer interpolates between them, so a 120 Hz brain can drive a
  60 Hz display without the two being coupled.
* **Telemetry** writes at 20 Hz, decimated inside :mod:`flybrain.telemetry`.

Everything the client sees is derived from here, so the app has exactly one
source of truth.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque

from .atlas import Atlas
from .connectome import (
    AssociativeMemory,
    BuiltinCircuitSource,
    ConnectomeRouter,
    JsonCircuitSource,
    Mutations,
)
from .env import CATALOG_BY_KEY, ENVIRONMENT_ORDER, Environment, World
from .fly import Fly, SocialDrive
from .neuro import EndocrineSystem, MODULATORS, preset_profiles
from .telemetry import Telemetry

__all__ = ["Kernel", "PRESET_SCENARIOS"]

STEP_HZ = 120.0
PUBLISH_HZ = 60.0
MAX_STEPS_PER_FRAME = 12
MAX_CATCHUP_S = 0.25


PRESET_SCENARIOS: dict[str, dict] = {
    "starving-minefield": {
        "label": "Starving Fly in a Minefield",
        "blurb": "Maximum hunger drive, maximum hazard. Watch it weigh a lethal stove against "
        "a sugar source it can smell but cannot reach safely.",
        "environment": "kitchen",
        "profile": "starving-minefield",
        "timeScale": 1.0,
        "blueprint": {
            "fuels": [("sucrose", -260.0, 34.0, 220.0), ("banana", -180.0, 40.0, -200.0)],
            "hazards": [("burner", 240.0, 26.0, 120.0), ("permethrin", 120.0, 40.0, 240.0),
                        ("swatter", 0.0, 40.0, -60.0)],
            "helpers": [],
        },
        "watch": "Risk tolerance climbs with hunger: it will cross the heat to reach sugar, and "
        "the moment hazard passes ~0.92 the giant fibre fires and it bolts.",
    },
    "dopamine-overdrive": {
        "label": "Dopamine Overdrive",
        "blurb": "Saturated reward signalling with almost no inhibition. Fast, persistent, "
        "reckless goal-seeking and little learning, because everything is already rewarding.",
        "environment": "biolab",
        "profile": "dopamine-overdrive",
        "timeScale": 1.0,
        "blueprint": {
            "fuels": [("sucrose", -190.0, 34.0, -120.0), ("vinegar", 190.0, 34.0, 130.0)],
            "hazards": [],
            "helpers": [("warm_patch", 0.0, 34.0, 0.0)],
        },
        "watch": "Motor gain roughly doubles and the fly re-decides less often, so it commits to "
        "a goal and sprints. Note the learned-valence bars staying flat: high DA alone does not "
        "create memory.",
    },
    "night-stupor": {
        "label": "Nighttime Sleep Stupor",
        "blurb": "Peak circadian rest pressure. Sleeping the day away with low reactivity.",
        "environment": "trashbin",
        "profile": "night-stupor",
        "timeScale": 1.0,
        "blueprint": {
            "fuels": [("banana", 60.0, 46.0, -160.0)],
            "hazards": [],
            "helpers": [("warm_patch", -120.0, 34.0, 60.0)],
        },
        "watch": "Reactivity collapses, so the same odour that once caused foraging now produces "
        "rest. Flick the clock slider up and watch it wake mid-session.",
    },
    "pavlovian": {
        "label": "Classical Pavlovian Conditioning",
        "blurb": "Peppermint is innately neutral. Pair it with a shock a few times and watch the "
        "fly learn to avoid it.",
        "environment": "biolab",
        "profile": "pavlovian",
        "timeScale": 1.0,
        "blueprint": {
            "fuels": [("banana", 250.0, 34.0, -160.0)],
            "hazards": [],
            "helpers": [("peppermint", -40.0, 40.0, 120.0)],
        },
        "conditioning": True,
        "watch": "Open the Memory panel and press 'Pair + shock'. Peppermint goes from 0 to "
        "aversive within about five trials if dopamine is up.",
    },
    "clean-slate": {
        "label": "Clean Slate",
        "blurb": "One fly, one empty arena, no stimuli but a warm patch. Build your own "
        "experiment from here.",
        "environment": "biolab",
        "profile": "baseline",
        "timeScale": 1.0,
        "blueprint": {"fuels": [], "hazards": [], "helpers": []},
        "watch": "Drop stimuli from the toolbelt and watch each modality light a different "
        "region of the brain.",
    },
}


class Kernel:
    """Owns the whole simulation and exposes a thread-safe command surface."""

    def __init__(
        self,
        atlas: Atlas,
        root: str = ".",
        environment: str = "kitchen",
        fly_count: int = 1,
        connectome_path: str | None = None,
    ) -> None:
        self.root = root
        self.atlas = atlas
        self.lock = threading.RLock()
        self.commands: deque = deque()
        self.atlas_regions = tuple(region.key for region in atlas.regions)

        self.world = World(environment)
        self.neuro = EndocrineSystem()
        self.mutations = Mutations()
        self.memory = AssociativeMemory(self.mutations)
        self.telemetry = Telemetry()
        self.social = SocialDrive()

        source = BuiltinCircuitSource()
        if connectome_path:
            external = JsonCircuitSource(connectome_path)
            if external.available():
                source = external
        self.connectome_source = source
        self.router = ConnectomeRouter(self.atlas_regions, source.load(), self.mutations)

        self.flies: list[Fly] = []
        self._spawn_flies(fly_count)

        self.time = 0.0
        self.step_index = 0
        self.paused = False
        self.time_scale = 1.0
        self.day_length_s = 180.0
        self.running = False
        self._thread: threading.Thread | None = None
        self._last_publish = 0.0
        self._stimulus_publish = 0.0
        self._slow_cache: dict = {}
        self.conditioning_enabled = False
        self.conditioning: dict = {
            "enabled": False,
            "cs": "peppermint",
            "us": "shock",
            "autoTrials": 0,
            "manualTrials": 0,
            "pending": None,
            "cooldown": 0.0,
        }
        # Two handoff buffers: the hot fields at 60 Hz and the heavy ones (environment
        # geometry, plume puffs) at 5 Hz. The HTTP threads only ever read these.
        self.last_snapshot: dict = {}
        self.last_slow: dict = {}
        self._slow_timer = 0.0
        self.performance = {"steps": 0, "hz": 0.0, "load": 0.0}
        self._frame_started = time.perf_counter()

        self.log(
            f"Kernel ready in {self.world.environment.label} at {STEP_HZ:.0f} Hz. "
            f"Atlas: {len(atlas.regions)} neuropil ROIs from "
            f"{'real VFB volumes' if not atlas.meta.get('synthetic') else 'a synthetic stand-in'}.",
            level="event",
            channel="system",
        )

    # -- setup ------------------------------------------------------------
    def _spawn_flies(self, count: int) -> None:
        count = max(1, min(10, int(count)))
        self.flies = []
        for index in range(count):
            angle = (index / count) * 2.0 * math.pi
            fly = Fly(
                id=f"fly-{index}",
                name=f"Fly {index + 1}",
                x=math.sin(angle) * 60.0,
                y=1.1,
                z=math.cos(angle) * 60.0,
                heading=angle,
            )
            fly.rng.seed(11 + index * 7)
            self.flies.append(fly)

    @property
    def primary(self) -> Fly:
        return self.flies[0]

    def log(self, message: str, level: str = "info", channel: str = "brain",
            fly: str | None = None, extra: dict | None = None) -> None:
        """Append to the event log, stamped with the current simulation time."""
        self.telemetry.log.add(self.time, message, level, channel, fly, extra)

    # -- thread control ---------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="flybrain-kernel", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        step = 1.0 / STEP_HZ
        publish = 1.0 / PUBLISH_HZ
        accumulator = 0.0
        previous = time.perf_counter()
        publish_timer = 0.0
        busy = 0.0
        window_start = previous
        steps_in_window = 0

        while self.running:
            now = time.perf_counter()
            frame = now - previous
            previous = now
            if frame > MAX_CATCHUP_S:
                # Machine was asleep or the process was starved: drop the backlog
                # instead of simulating minutes of stray time at once.
                frame = step
            accumulator += frame
            steps = 0
            started = time.perf_counter()
            while accumulator >= step and steps < MAX_STEPS_PER_FRAME:
                if not self.paused:
                    self._step(step * self.time_scale)
                    steps_in_window += 1
                accumulator -= step
                steps += 1
            if accumulator > step * MAX_STEPS_PER_FRAME:
                accumulator = 0.0
            busy += time.perf_counter() - started

            publish_timer += frame
            if publish_timer >= publish:
                publish_timer = 0.0
                self._publish()

            self._slow_timer += frame
            if self._slow_timer >= 0.2:
                self._slow_timer = 0.0
                self._publish_slow()

            if now - window_start >= 0.5:
                elapsed = now - window_start
                self.performance["hz"] = round(steps_in_window / elapsed, 1)
                self.performance["load"] = round(busy / elapsed, 4)
                busy = 0.0
                steps_in_window = 0
                window_start = now

            sleep_for = step - (time.perf_counter() - now)
            if sleep_for > 0.0002:
                time.sleep(sleep_for)

    # -- one physics step -------------------------------------------------
    def _step(self, dt: float) -> None:
        self._drain_commands()
        self.time += dt
        self.step_index += 1

        # Threats aim at the primary fly, so the world needs a focus point. Set it
        # before stepping the world so a threat's motion this tick is aimed correctly.
        self.world.focus = self.primary.position
        self.world.step(dt)

        events: list[dict] = []
        for fly in self.flies:
            fly.update(
                dt,
                self.world,
                self.neuro,
                self.router,
                self.memory,
                self.mutations,
                self.social,
                self.flies,
                events,
            )

        primary = self.primary
        # The endocrine system integrates last, so it reacts to what just happened.
        activity = min(1.0, primary.speed / 380.0) + (0.4 if primary.is_airborne() else 0.0)
        feeding = 0.6 if primary.mode == "FEED" else 0.0
        self.neuro.step(dt, min(1.0, activity), feeding, primary.last_threat_score)

        self._process_events(events)
        self._conditioning_tick(dt)
        self.memory.decay(dt)

        self._record(dt)

    def _record(self, dt: float) -> None:
        primary = self.primary
        values = {
            "speed": primary.speed,
            "hunger": self.neuro.state.hunger,
            "fear": self.neuro.state.fear,
            "odor": primary.last_attract + primary.last_aversive,
            "energy": self.neuro.state.energy,
            "arousal": self.neuro.state.arousal,
        }
        for spec in MODULATORS:
            values[spec.key] = self.neuro.level(spec.key)
        for key in self.atlas_regions:
            values[key] = max(0.0, self.router.signal(key))
        self.telemetry.record(self.time, values, dt)
        self.telemetry.stats["distanceMm"] = round(
            sum(f.distance_travelled for f in self.flies), 1
        )
        self.telemetry.stats["escapes"] = sum(f.escape_events for f in self.flies)
        self.telemetry.stats["groomingEvents"] = sum(f.grooming_events for f in self.flies)

    # -- events and learning ----------------------------------------------
    def _process_events(self, events: list[dict]) -> None:
        for event in events:
            if event.get("kind") == "mode":
                level = "alert" if event["to"] in ("ESCAPE",) else "event"
                if event["to"] == "FEED":
                    self.telemetry.stats["feeds"] += 1
                self.log(event["message"], level=level, channel="behaviour", fly=event.get("fly"))
            elif event.get("kind") == "escape":
                self.log(
                    event["message"],
                    level="alert",
                    channel="reflex",
                    fly=event.get("fly"),
                    extra={"pathway": "OL -> GF -> CC", "latencyMs": round(1000.0 / STEP_HZ * 4, 1)},
                )

    def _conditioning_tick(self, dt: float) -> None:
        """Drive both explicit pairing and incidental learning.

        Explicit pairing (the Pavlovian preset) is a controlled trial. Incidental
        learning is the interesting one: whenever a real shock or a real sugar meal
        happens, whatever the fly is currently smelling gets paired with it. That is
        classical conditioning as it happens in the world, not in a protocol.
        """
        conditioning = self.conditioning
        conditioning["cooldown"] = max(0.0, conditioning["cooldown"] - dt)

        pending = conditioning.get("pending")
        if pending and conditioning["cooldown"] <= 0.0:
            conditioning["pending"] = None
            self._apply_pairing(pending[0], pending[1], explicit=True)

        if not conditioning["enabled"]:
            return

        for fly in self.flies:
            if not fly.attended_odor:
                continue
            threat = fly.last_threat_score
            if threat > 0.9 and conditioning["cooldown"] <= 0.0:
                conditioning["cooldown"] = 2.5
                self._apply_pairing(fly.attended_odor[0], -1.0, explicit=False)
            elif fly.mode == "FEED" and conditioning["cooldown"] <= 0.0:
                conditioning["cooldown"] = 2.5
                self._apply_pairing(fly.attended_odor[0], 1.0, explicit=False)

    def _apply_pairing(self, odor_key: str, us_valence: float, explicit: bool) -> None:
        spec = CATALOG_BY_KEY.get(odor_key)
        label = spec.label if spec else odor_key
        learning_rate = self.neuro.learning_rate(self.mutations.acquisition_gain)
        before, after = self.memory.condition(odor_key, label, us_valence, learning_rate)
        self.telemetry.stats["conditioningTrials"] += 1
        if explicit:
            self.conditioning["manualTrials"] += 1
        else:
            self.conditioning["autoTrials"] += 1

        if self.mutations.rutabaga:
            self.log(
                f"Conditioning trial: {label} paired with "
                f"{'shock' if us_valence < 0 else 'sugar'} - rutabaga mutant, "
                f"acquisition gain is 0, no memory formed",
                level="learn",
                channel="memory",
            )
            return
        direction = "aversive" if us_valence < 0 else "attractive"
        self.log(
            f"Memory: {label} paired with {'shock' if us_valence < 0 else 'sugar'} "
            f"-> valence {before:+.2f} to {after:+.2f} "
            f"(now {direction}, trial {self.memory.entry(odor_key, label).trials})",
            level="learn",
            channel="memory",
            extra={"cs": odor_key, "us": us_valence, "before": round(before, 3), "after": round(after, 3)},
        )

    # -- publishing -------------------------------------------------------
    def _publish(self) -> None:
        snapshot = self.snapshot()
        with self.lock:
            self.last_snapshot = snapshot

    def _publish_slow(self) -> None:
        """Publish environment geometry and plumes, which change far more slowly."""
        payload = {
            "world": {
                "key": self.world.environment.key,
                "label": self.world.environment.label,
                "time": round(self.world.time, 3),
                "focus": [round(v, 2) for v in self.world.focus],
                "stimuli": [stimulus.to_json() for stimulus in self.world.stimuli],
            }
        }
        with self.lock:
            self.last_slow = payload

    def snapshot(self) -> dict:
        """Build the hot client payload. Called at 60 Hz, so it stays allocation-light."""
        primary = self.primary
        endocrine = self.neuro.to_json()
        router = self.router.to_json()

        # Regional activation is what the brain HUD raymarches, so it travels on the
        # hot path while the heavier environment payload is throttled.
        region_activation = {
            key: round(max(0.0, self.router.signal(key)), 4) for key in self.atlas_regions
        }

        payload = {
            "t": round(self.time, 3),
            "step": self.step_index,
            "paused": self.paused,
            "timeScale": self.time_scale,
            "performance": dict(self.performance),
            "endocrine": endocrine,
            "router": {
                "activation": router["activation"],
                "traffic": router["traffic"],
                "readouts": router["readouts"],
                "circuit": router["circuit"],
            },
            "regionActivation": region_activation,
            "flies": [fly.to_json() for fly in self.flies],
            "primary": primary.to_json(),
            "memory": self.memory.to_json(),
            "mutations": self.mutations.to_json(),
            "conditioning": {
                "enabled": self.conditioning["enabled"],
                "cs": self.conditioning["cs"],
                "us": self.conditioning["us"],
                "autoTrials": self.conditioning["autoTrials"],
                "manualTrials": self.conditioning["manualTrials"],
            },
            "stats": dict(self.telemetry.stats),
            "flyCount": len(self.flies),
        }
        return payload

    # -- command handling -------------------------------------------------
    def post(self, name: str, payload: dict | None = None) -> None:
        """Queue a command from any thread."""
        self.commands.append((name, payload or {}))

    def _drain_commands(self) -> None:
        with self.lock:
            commands = list(self.commands)
            self.commands.clear()
        for name, payload in commands:
            try:
                self._apply(name, payload)
            except Exception as error:  # pragma: no cover - keep the kernel alive
                self.log(f"Command {name!r} failed: {error}", level="alert", channel="system")

    def _apply(self, name: str, payload: dict) -> None:
        handler = getattr(self, f"_cmd_{name}", None)
        if handler is None:
            self.log(f"Unknown command {name!r}", level="alert", channel="system")
            return
        handler(payload)

    # individual commands
    def _cmd_set_slider(self, payload: dict) -> None:
        key = payload.get("key", "")
        self.neuro.set_slider(key, float(payload.get("value", 0.0)))
        spec = next((s for s in MODULATORS if s.key == key), None)
        if spec:
            self.log(
                f"{spec.label} set to {float(payload.get('value', 0.0)):.0%} - {spec.effect}",
                level="event",
                channel="control",
            )

    def _cmd_apply_profile(self, payload: dict) -> None:
        name = payload.get("name", "baseline")
        profiles = preset_profiles()
        profile = profiles.get(name)
        if profile is None:
            return
        self.neuro.apply_profile(profile)
        self.log(f"Endocrine profile '{name}' applied", level="event", channel="control")

    def _cmd_set_mutation(self, payload: dict) -> None:
        key = payload.get("key", "")
        value = bool(payload.get("value", False))
        if hasattr(self.mutations, key):
            setattr(self.mutations, key, value)
            self.log(
                f"Mutation {key} {'enabled' if value else 'cleared'}",
                level="event",
                channel="control",
            )

    def _cmd_load_environment(self, payload: dict) -> None:
        key = payload.get("key", "kitchen")
        if key not in ENVIRONMENT_ORDER:
            return
        self.world.load_environment(key)
        self.telemetry.reset()
        for fly in self.flies:
            fly.x = 0.0
            fly.z = 0.0
            fly.y = 1.1
            fly.speed = 0.0
            fly.mode = "REST"
            fly.footprint.clear()
        self.log(
            f"Environment loaded: {self.world.environment.label} - {self.world.environment.notes}",
            level="event",
            channel="system",
        )

    def _cmd_add_stimulus(self, payload: dict) -> None:
        stim = self.world.add(
            payload.get("key", ""),
            float(payload.get("x", 0.0)),
            float(payload.get("y", 30.0)),
            float(payload.get("z", 0.0)),
        )
        if stim:
            self.log(
                f"Placed {stim.spec.label} at {stim.x:.0f},{stim.y:.0f},{stim.z:.0f} mm - "
                f"{stim.spec.effect}",
                level="event",
                channel="control",
            )

    def _cmd_remove_stimulus(self, payload: dict) -> None:
        if self.world.remove(payload.get("id", "")):
            self.log("Removed a stimulus", level="event", channel="control")

    def _cmd_move_stimulus(self, payload: dict) -> None:
        self.world.move(
            payload.get("id", ""),
            float(payload.get("x", 0.0)),
            float(payload.get("y", 30.0)),
            float(payload.get("z", 0.0)),
        )

    def _cmd_set_param(self, payload: dict) -> None:
        self.world.set_param(payload.get("id", ""), payload.get("name", ""),
                             float(payload.get("value", 0.0)))

    def _cmd_clear_stimuli(self, payload: dict) -> None:
        self.world.clear_stimuli(keep_sources=bool(payload.get("keepSources", True)))
        self.log("Stimuli cleared", level="event", channel="control")

    def _cmd_set_paused(self, payload: dict) -> None:
        self.paused = bool(payload.get("value", False))
        self.log("Simulation paused" if self.paused else "Simulation resumed",
                 level="event", channel="system")

    def _cmd_set_time_scale(self, payload: dict) -> None:
        self.time_scale = max(0.1, min(6.0, float(payload.get("value", 1.0))))
        self.log(f"Time scale {self.time_scale:.2f}x", level="event", channel="system")

    def _cmd_set_day_length(self, payload: dict) -> None:
        self.day_length_s = max(20.0, min(3600.0, float(payload.get("value", 180.0))))
        self.neuro.day_length_s = self.day_length_s
        self.log(f"Day length {self.day_length_s:.0f} s", level="event", channel="system")

    def _cmd_set_fly_count(self, payload: dict) -> None:
        count = int(payload.get("value", 1))
        if count == len(self.flies):
            return
        self._spawn_flies(count)
        self.log(
            f"{'Swarming mode: ' + str(count) + ' flies' if count > 1 else 'Single-fly inspection mode'}",
            level="event",
            channel="system",
        )

    def _cmd_condition(self, payload: dict) -> None:
        """Queue one explicit CS/US pairing (the Pavlovian 'train' button)."""
        cs = payload.get("cs", self.conditioning["cs"])
        us = payload.get("us", "shock")
        self.conditioning["cs"] = cs
        self.conditioning["us"] = us
        valence = -1.0 if us == "shock" else 1.0
        self.conditioning["pending"] = (cs, valence)
        self.conditioning["cooldown"] = 0.4

    def _cmd_set_conditioning(self, payload: dict) -> None:
        self.conditioning["enabled"] = bool(payload.get("enabled", False))
        if "cs" in payload:
            self.conditioning["cs"] = payload["cs"]
        if "us" in payload:
            self.conditioning["us"] = payload["us"]
        state = "enabled" if self.conditioning["enabled"] else "disabled"
        self.log(f"Automatic conditioning {state}", level="event", channel="memory")

    def _cmd_forget(self, payload: dict) -> None:
        key = payload.get("key")
        if key and key in self.memory.entries:
            del self.memory.entries[key]
            self.log(f"Cleared memory for {key}", level="learn", channel="memory")
        elif not key:
            self.memory.entries.clear()
            self.log("All learned odour values cleared", level="learn", channel="memory")

    def _cmd_deliver(self, payload: dict) -> None:
        """Fire an unconditioned stimulus at the fly's current position."""
        kind = payload.get("kind", "shock")
        if kind == "shock":
            self.neuro.pulse("OA", 0.5)
            self.neuro.state.fear = 1.0
            for fly in self.flies:
                if fly.attended_odor and self.conditioning["enabled"]:
                    self._apply_pairing(fly.attended_odor[0], -1.0, explicit=True)
            self.log("Electric shock delivered", level="alert", channel="memory")
        else:
            self.neuro.pulse("DA", 0.6)
            self.neuro.state.energy = min(1.0, self.neuro.state.energy + 0.15)
            for fly in self.flies:
                if fly.attended_odor and self.conditioning["enabled"]:
                    self._apply_pairing(fly.attended_odor[0], 1.0, explicit=True)
            self.log("Sugar reward delivered", level="learn", channel="memory")

    def _cmd_preset(self, payload: dict) -> None:
        preset = PRESET_SCENARIOS.get(payload.get("name", ""))
        if preset is None:
            return
        self._cmd_load_environment({"key": preset["environment"]})
        self._cmd_apply_profile({"name": preset["profile"]})
        self.time_scale = float(preset.get("timeScale", 1.0))
        self.world.clear_stimuli(keep_sources=False)
        blueprint = preset.get("blueprint", {})
        for key, x, y, z in blueprint.get("fuels", []):
            self.world.add(key, x, y, z)
        for key, x, y, z in blueprint.get("hazards", []):
            self.world.add(key, x, y, z)
        for key, x, y, z in blueprint.get("helpers", []):
            self.world.add(key, x, y, z)
        if not blueprint.get("helpers"):
            self.world.add("warm_patch", -200.0, 34.0, 200.0)
        self.neuro.set_slider("CLOCK", 0.6)
        self.conditioning["enabled"] = bool(preset.get("conditioning", False))
        self.paused = False
        self.log(
            f"Scenario loaded: {preset['label']} - {preset['watch']}",
            level="event",
            channel="system",
        )

    def _cmd_reset(self, payload: dict) -> None:
        self.telemetry.reset()
        self.memory.entries.clear()
        self.mutations = Mutations()
        self.memory.mutations = self.mutations
        self.router.mutations = self.mutations
        self.neuro = EndocrineSystem(self.day_length_s)
        self.time = 0.0
        self.step_index = 0
        self.world.load_environment(self.world.environment.key)
        self._spawn_flies(int(payload.get("flies", len(self.flies))))
        self.log("Session reset", level="event", channel="system")

    def _cmd_export(self, payload: dict) -> None:
        written = self.telemetry.write_exports(
            payload.get("directory", "exports"),
            stem=payload.get("stem", "flybrain-session"),
            extra={
                "environment": self.world.environment.key,
                "endocrine": self.neuro.to_json(),
                "mutations": self.mutations.to_json(),
                "memory": self.memory.to_json(),
                "atlas": {"shape": list(self.atlas.shape), "meta": self.atlas.meta},
            },
        )
        self.log(
            "Export written: " + ", ".join(written),
            level="event",
            channel="system",
            extra={"files": written},
        )

    def _cmd_export_profile(self, payload: dict) -> None:
        import json

        directory = payload.get("directory", "profiles")
        name = payload.get("name", "endocrine-profile")
        import os

        os.makedirs(os.path.join(self.root, directory), exist_ok=True)
        path = os.path.join(self.root, directory, f"{name}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.neuro.targets(), handle, indent=2)
        self.log(f"Endocrine profile saved to {path}", level="event", channel="system")

    def _cmd_load_profile(self, payload: dict) -> None:
        import json
        import os

        path = os.path.join(self.root, payload.get("path", ""))
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            self.neuro.apply_profile(json.load(handle))
        self.log(f"Endocrine profile loaded from {path}", level="event", channel="system")

    # -- teardown ---------------------------------------------------------
    def shutdown(self) -> None:
        self.stop()
