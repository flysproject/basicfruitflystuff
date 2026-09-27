"""Environment, stimuli and the physical fields the fly senses.

Coordinate system: millimetres, right-handed, **y is up**. The arena is a
counter-scale volume (~900 x 420 x 600 mm), the fly is 3.2 mm long. Nothing is
faked for visibility: the fly really is 3 mm in a 1 m room, and the camera zooms.

Odour is modelled as a **puff cloud**, which is both cheap and correct:

* Each source emits puffs on an interval. A puff drifts with the wind, grows by
  diffusion, and is culled after its lifetime.
* Concentration at a point is the sum of Gaussian kernels over the live puffs,
  plus a small persistent kernel at the source itself (the food really does smell
  locally).

That choice buys three things. It is O(live puffs) per query with no grid and no
diffusion solver, so the 120 Hz kernel can afford it in pure Python. It produces
*intermittent* plumes, so odour-guided navigation has to cast and re-acquire
instead of following a smooth gradient -- which is what real flies do. And because
a puff is just an ellipsoid with a strength, the renderer draws exactly the field
the fly smells, with no second implementation to drift out of sync.

Thermal zones and light use the same analytic approach for the same reason.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .neuro import clamp

__all__ = [
    "Stimulus",
    "Puff",
    "Environment",
    "ENVIRONMENTS",
    "ENVIRONMENT_ORDER",
    "STIMULUS_CATALOG",
    "World",
]

# --------------------------------------------------------------------------
# Stimulus catalogue: what the toolbelt can drop into the world
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StimulusSpec:
    kind: str
    key: str
    label: str
    color: str
    blurb: str
    effect: str
    icon: str
    params: dict = field(default_factory=dict)


STIMULUS_CATALOG: tuple[StimulusSpec, ...] = (
    # -- odours -----------------------------------------------------------
    StimulusSpec(
        kind="odor",
        key="banana",
        label="Rotten banana",
        color="#f6d365",
        icon="banana",
        blurb="Strong fermenting-fruit odour. Ethyl acetate and ethanol blend that signals "
        "ripe food to a foraging fly.",
        effect="Strong attraction. Peak plume strength, emitter interval 1.4 s.",
        params={"sigma": 42.0, "strength": 1.0, "valence": 0.95, "interval": 1.4, "life": 9.0,
                 "broad_sigma": 300.0, "broad_strength": 0.40},
    ),
    StimulusSpec(
        kind="odor",
        key="sucrose",
        label="Sucrose droplet",
        color="#ffe9a8",
        icon="drop",
        blurb="Pure sugar water. Tastants matter more than odour here: the fly mostly finds "
        "it by contact once nearby.",
        effect="Attraction with a strong local contact reward; drives proboscis extension.",
        params={"sigma": 22.0, "strength": 0.55, "valence": 0.9, "interval": 2.6, "life": 8.0,
                 "broad_sigma": 120.0, "broad_strength": 0.22},
    ),
    StimulusSpec(
        kind="odor",
        key="vinegar",
        label="Apple cider vinegar",
        color="#c98b5a",
        icon="flask",
        blurb="Acetic acid. Flies are attracted to fermenting substrates, so vinegar reads as "
        "'food nearby with slightly wrong chemistry'.",
        effect="Moderate attraction, weaker and shorter-lived than fruit.",
        params={"sigma": 36.0, "strength": 0.7, "valence": 0.45, "interval": 1.8, "life": 6.0,
                 "broad_sigma": 260.0, "broad_strength": 0.30},
    ),
    StimulusSpec(
        kind="odor",
        key="ethanol",
        label="Ethanol vapour",
        color="#9fb4c7",
        icon="flask",
        blurb="A fermentation product that flies tolerate well and can even prefer.",
        effect="Mild attraction that also mildly sedates at high concentration.",
        params={"sigma": 46.0, "strength": 0.5, "valence": 0.3, "interval": 2.2, "life": 7.0,
                 "broad_sigma": 280.0, "broad_strength": 0.26},
    ),
    StimulusSpec(
        kind="odor",
        key="citronella",
        label="Citronella",
        color="#8fd14f",
        icon="leaf",
        blurb="Plant volatile that insects read as a plant, not a fruit: no fermentation, no "
        "sugar.",
        effect="Strong avoidance. Raises innate aversion and suppresses feeding.",
        params={"sigma": 40.0, "strength": 0.85, "valence": -0.85, "interval": 1.6, "life": 8.0,
                 "broad_sigma": 250.0, "broad_strength": 0.32},
    ),
    StimulusSpec(
        kind="odor",
        key="permethrin",
        label="Permethrin",
        color="#ff6b6b",
        icon="skull",
        blurb="Pyrethroid insecticide. Neurotoxic to insects, and flies show rapid learned and "
        "innate avoidance.",
        effect="Strongest avoidance in the catalogue; also a punishing unconditioned stimulus.",
        params={"sigma": 44.0, "strength": 1.0, "valence": -1.0, "interval": 1.5, "life": 9.0,
                 "broad_sigma": 240.0, "broad_strength": 0.34},
    ),
    StimulusSpec(
        kind="odor",
        key="peppermint",
        label="Peppermint (neutral)",
        color="#a5f3fc",
        icon="spark",
        blurb="A plant odour with no innate value to a fly. It is the classic conditioned "
        "stimulus precisely because the fly starts out indifferent.",
        effect="No innate valence: only useful as the CS in a conditioning pairing.",
        params={"sigma": 45.0, "strength": 0.8, "valence": 0.0, "interval": 1.7, "life": 9.0,
                 "broad_sigma": 240.0, "broad_strength": 0.30},
    ),
    # -- light ------------------------------------------------------------
    StimulusSpec(
        kind="light",
        key="spotlight",
        label="Movable spotlight",
        color="#fff4c2",
        icon="light",
        blurb="A bright point source. Flies show strong positive phototaxis toward it.",
        effect="Drives phototaxis and, when swung fast, the loom detectors.",
        params={"intensity": 1.6, "radius": 900.0, "tone": "#fff3c4"},
    ),
    StimulusSpec(
        kind="light",
        key="lamp",
        label="Ceiling lamp",
        color="#ffffff",
        icon="lamp",
        blurb="Broad overhead illumination. Sets the ambient light level for the arena.",
        effect="Raises overall contrast for the optic lobe and dampens night behaviour.",
        params={"intensity": 1.1, "radius": 3000.0, "tone": "#ffffff"},
    ),
    StimulusSpec(
        kind="light",
        key="sunbeam",
        label="Sunbeam",
        color="#ffe9b0",
        icon="sun",
        blurb="Collimated sunlight. Directional, so it also creates a strong thermal gradient.",
        effect="High visual contrast plus a gentle warming gradient.",
        params={"intensity": 2.0, "radius": 8000.0, "tone": "#ffeeb8"},
    ),
    # -- thermal ----------------------------------------------------------
    StimulusSpec(
        kind="thermal",
        key="burner",
        label="Stove burner",
        color="#ff5722",
        icon="flame",
        blurb="A genuinely lethal surface for a fly: above roughly 40 C the fly shows strong "
        "avoidance and rapid escape.",
        effect="Heat threat. Above ~38 C the fly refuses to enter and bolts if it does.",
        params={"delta_c": 62.0, "sigma": 90.0, "kind": "hot"},
    ),
    StimulusSpec(
        kind="thermal",
        key="ice",
        label="Ice cube",
        color="#7dd3fc",
        icon="snow",
        blurb="Cold avoidance. Flies prefer roughly 24-27 C and turn away from sharp cold.",
        effect="Cold avoidance zone, sigma 70 mm.",
        params={"delta_c": -18.0, "sigma": 70.0, "kind": "cold"},
    ),
    StimulusSpec(
        kind="thermal",
        key="warm_patch",
        label="Optimal warmth",
        color="#fbbf24",
        icon="sun",
        blurb="A 25-27 C patch, the thermotaxis optimum for Drosophila.",
        effect="Attractive thermal zone: the fly will settle and groom here.",
        params={"delta_c": 2.5, "sigma": 150.0, "kind": "warm"},
    ),
    # -- threats ----------------------------------------------------------
    StimulusSpec(
        kind="threat",
        key="swatter",
        label="Fly swatter",
        color="#e2e8f0",
        icon="swatter",
        blurb="A fast-moving disc roughly 100x the fly's body length. Triggers the lobula giant "
        "movement detector -- the classic looming response.",
        effect="Swinging threat: looming rate drives escape takeoff within ~30 ms.",
        params={"radius": 55.0, "speed": 900.0, "swing": 260.0, "danger": 1.0,
                 "track_speed": 55.0},
    ),
    StimulusSpec(
        kind="threat",
        key="hand",
        label="Human hand",
        color="#f5c8a8",
        icon="hand",
        blurb="Slow, large, and warm. Flies habituate to slow looming less well than to fast "
        "looming but it is harder to outrun a hand.",
        effect="Slow large threat with a thermal gradient attached.",
        params={"radius": 95.0, "speed": 380.0, "swing": 420.0, "danger": 0.8,
                 "dip": 0.30, "hover": 70.0, "track_speed": 34.0},
    ),
    StimulusSpec(
        kind="threat",
        key="predator",
        label="Predator motion",
        color="#ef4444",
        icon="paw",
        blurb="Fast erratic approach. The least predictable threat and the highest escape "
        "priority.",
        effect="Fast threatening motion with an erratic trajectory.",
        params={"radius": 70.0, "speed": 1300.0, "swing": 300.0, "danger": 1.2,
                 "track_speed": 46.0},
    ),
)

CATALOG_BY_KEY = {spec.key: spec for spec in STIMULUS_CATALOG}


# --------------------------------------------------------------------------
# Live stimuli
# --------------------------------------------------------------------------


@dataclass
class Puff:
    """One advected, diffusing odour cloud."""

    x: float
    y: float
    z: float
    sigma: float
    strength: float
    age: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    def to_json(self) -> dict:
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "z": round(self.z, 2),
            "sigma": round(self.sigma, 2),
            "strength": round(self.strength, 4),
        }


@dataclass
class Stimulus:
    """A placed stimulus, whether dropped by the user or part of a preset."""

    id: str
    key: str
    kind: str
    x: float
    y: float
    z: float
    params: dict = field(default_factory=dict)
    puffs: list[Puff] = field(default_factory=list)
    emit_timer: float = 0.0
    phase: float = 0.0
    active: bool = True
    # Threat bookkeeping. vel_* is the threat's velocity in mm/s, derived from the
    # per-frame displacement. Storing an explicit velocity rather than letting the
    # consumer take differences is deliberate: an earlier version differenced the
    # positions inside the sensor and silently left the result in mm *per frame*,
    # which made looming depend on the tick rate and let fast threats trigger the
    # escape reflex while a slow, large hand never triggered it at all.
    last_x: float = 0.0
    last_y: float = 0.0
    last_z: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_z: float = 0.0
    loom_rate: float = 0.0
    angular_size: float = 0.0

    @property
    def spec(self) -> StimulusSpec:
        return CATALOG_BY_KEY[self.key]

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_json(self) -> dict:
        payload = {
            "id": self.id,
            "key": self.key,
            "kind": self.kind,
            "label": self.spec.label,
            "color": self.spec.color,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "z": round(self.z, 2),
        }
        if self.kind == "odor":
            payload["puffs"] = [p.to_json() for p in self.puffs]
            payload["strength"] = self.params.get("strength", 1.0)
            payload["sigma"] = self.params.get("sigma", 40.0)
        if self.kind == "thermal":
            payload["deltaC"] = self.params.get("delta_c", 0.0)
            payload["sigma"] = self.params.get("sigma", 80.0)
            payload["thermalKind"] = self.params.get("kind", "warm")
        if self.kind == "light":
            payload["intensity"] = self.params.get("intensity", 1.0)
            payload["tone"] = self.params.get("tone", "#ffffff")
        if self.kind == "threat":
            payload["radius"] = self.params.get("radius", 60.0)
            payload["loomRate"] = round(self.loom_rate, 4)
            payload["angularSize"] = round(self.angular_size, 4)
        return payload


# --------------------------------------------------------------------------
# Environments
# --------------------------------------------------------------------------

# Height a standing fly's body centre sits above the surface it is standing on.
# World only needs it to seed the threat focus before the kernel takes over.
WALK_HEIGHT = 1.1

BASE_TEMP_C = 24.0
OPTIMAL_TEMP_C = 26.0
LETHAL_TEMP_C = 40.0


@dataclass(frozen=True)
class Environment:
    key: str
    label: str
    blurb: str
    size: tuple[float, float, float]
    wind: tuple[float, float, float]
    ambient_temp_c: float
    ambient_light: float
    props: tuple[dict, ...]
    ambient_odors: tuple[tuple[str, float, float, float], ...] = ()
    notes: str = ""
    skybox: str = "#0b1220"
    fog: float = 0.0006

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "blurb": self.blurb,
            "size": list(self.size),
            "wind": list(self.wind),
            "ambientTempC": self.ambient_temp_c,
            "ambientLight": self.ambient_light,
            "props": list(self.props),
            "notes": self.notes,
            "skybox": self.skybox,
            "fog": self.fog,
        }


ENVIRONMENTS: dict[str, Environment] = {
    "kitchen": Environment(
        key="kitchen",
        label="Human Kitchen",
        blurb="Warm countertop with a rotting fruit bowl, a wet sink and a running ceiling "
        "fan. The fan is the dominant physical force: it advects every odour plume.",
        size=(900.0, 420.0, 600.0),
        wind=(0.0, -140.0, 60.0),
        ambient_temp_c=24.0,
        ambient_light=0.55,
        skybox="#1a1410",
        fog=0.0009,
        notes="Fan-driven wind creates intermittent plumes, so the fly must cast to re-acquire "
        "the odour.",
        props=(
            {"type": "counter", "x": 0, "y": 0, "z": 0, "sx": 900, "sy": 24, "sz": 600},
            {"type": "fruit_bowl", "x": -210, "y": 24, "z": -120, "r": 110},
            {"type": "rotting_fruit", "x": -210, "y": 56, "z": -120, "r": 46, "color": "#8a6b2f"},
            {"type": "fruit_chunk", "x": -262, "y": 24, "z": -158, "r": 30, "color": "#93702a"},
            {"type": "sink", "x": 240, "y": 6, "z": 130, "sx": 320, "sy": 40, "sz": 240},
            {"type": "drain", "x": 240, "y": 26, "z": 130, "r": 34},
            {"type": "ceiling_fan", "x": 0, "y": 420, "z": 0, "r": 190, "rpm": 42},
            {"type": "lamp", "x": 300, "y": 400, "z": -220, "r": 70, "color": "#fff3c4"},
            {"type": "mug", "x": 120, "y": 24, "z": -240, "r": 42},
            {"type": "wall", "x": 0, "y": 210, "z": -300, "sx": 900, "sy": 420, "sz": 8},
        ),
        # The odour source sits slightly downwind of the fruit bowl, so the plume
        # is advected onto the countertop where a walking fly can actually reach it.
        ambient_odors=(("banana", -250.0, 30.0, -150.0),),
    ),
    "garden": Environment(
        key="garden",
        label="Outdoor Garden",
        blurb="Open ground under a low sun, with flowers, drifting wind and a damp puddle. "
        "Long sightlines make vision the dominant sense here.",
        size=(1000.0, 400.0, 700.0),
        wind=(90.0, 6.0, -40.0),
        ambient_temp_c=22.0,
        ambient_light=1.0,
        skybox="#3b6ea5",
        fog=0.0011,
        notes="Strong directional light and steady wind; odour plumes stretch far downwind.",
        props=(
            {"type": "ground", "x": 0, "y": 0, "z": 0, "sx": 1000, "sy": 20, "sz": 700},
            {"type": "flowers", "x": -260, "y": 20, "z": -160, "r": 130, "count": 9},
            {"type": "flowers", "x": 250, "y": 20, "z": 190, "r": 120, "count": 7},
            {"type": "puddle", "x": 150, "y": 22, "z": -220, "r": 150},
            {"type": "rock", "x": -60, "y": 20, "z": 240, "r": 70},
            {"type": "sun", "x": -700, "y": 700, "z": -500},
            {"type": "grass", "x": 0, "y": 20, "z": 0, "sx": 1000, "sz": 700, "count": 260},
        ),
        ambient_odors=(("citronella", 250.0, 60.0, 190.0),),
    ),
    "trashbin": Environment(
        key="trashbin",
        label="Trash Bin",
        blurb="A dark cylindrical interior packed with decaying matter. High odour "
        "concentration, tight geometry, no long sightlines.",
        size=(420.0, 520.0, 420.0),
        wind=(0.0, -18.0, 0.0),
        ambient_temp_c=26.5,
        ambient_light=0.12,
        skybox="#0a0a0a",
        fog=0.0042,
        notes="Every odour is strong and everything is close: this is a contact-range "
        "environment dominated by olfaction.",
        props=(
            {"type": "bin_wall", "x": 0, "y": 260, "z": 0, "r": 210, "h": 520},
            {"type": "bin_floor", "x": 0, "y": 0, "z": 0, "r": 210},
            {"type": "trash_pile", "x": -100, "y": 30, "z": -60, "r": 100, "color": "#4a4a3a"},
            {"type": "trash_pile", "x": 110, "y": 20, "z": 90, "r": 80, "color": "#5a4a34"},
            {"type": "trash_pile", "x": 20, "y": 90, "z": -140, "r": 70, "color": "#3f3a30"},
            {"type": "rotten_fruit", "x": 60, "y": 46, "z": -160, "r": 52, "color": "#7a5c22"},
        ),
        ambient_odors=(
            ("banana", 60.0, 46.0, -160.0),
            ("vinegar", -100.0, 40.0, -60.0),
        ),
    ),
    "biolab": Environment(
        key="biolab",
        label="Clean Bio-Lab",
        blurb="A controlled bench with sucrose-agar petri dishes and a printed grid arena. "
        "Closest thing to a real Drosophila assay plate.",
        size=(760.0, 320.0, 520.0),
        wind=(0.0, -4.0, 0.0),
        ambient_temp_c=25.0,
        ambient_light=0.75,
        skybox="#0d1524",
        fog=0.0004,
        notes="Near-still air, so odour stays local. Best environment for conditioning "
        "experiments.",
        props=(
            {"type": "bench", "x": 0, "y": 0, "z": 0, "sx": 760, "sy": 22, "sz": 520},
            {"type": "arena_grid", "x": 0, "y": 22, "z": 0, "sx": 700, "sy": 2, "sz": 460},
            {"type": "petri", "x": -190, "y": 22, "z": -120, "r": 90, "color": "#d8c98a"},
            {"type": "petri", "x": 150, "y": 22, "z": 130, "r": 90, "color": "#c9d8a0"},
            {"type": "petri", "x": 210, "y": 22, "z": -150, "r": 70, "color": "#d8c98a"},
            {"type": "lamp_bar", "x": 0, "y": 300, "z": 0, "sx": 620, "sz": 40},
            {"type": "microscope", "x": -260, "y": 22, "z": 180, "r": 60},
        ),
        ambient_odors=(("sucrose", -190.0, 34.0, -120.0),),
    ),
}

ENVIRONMENT_ORDER = ("kitchen", "garden", "trashbin", "biolab")


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------


class World:
    """The physical world: stimuli, their fields, wind, and threat geometry."""

    MAX_PUFFS_PER_SOURCE = 9
    MAX_PUFFS_TOTAL = 110

    def __init__(self, environment: str = "kitchen", seed: int = 7) -> None:
        self.random = random.Random(seed)
        self.environment = ENVIRONMENTS.get(environment, ENVIRONMENTS["kitchen"])
        self.stimuli: list[Stimulus] = []
        self.time = 0.0
        self._next_id = 1
        # Where the threats aim. The kernel sets this to the primary fly each step,
        # which keeps World free of any dependency on the agent classes.
        self.focus: tuple[float, float, float] = (0.0, WALK_HEIGHT, 0.0)
        self.load_environment(self.environment.key)

    # -- lifecycle --------------------------------------------------------
    def load_environment(self, key: str) -> None:
        self.environment = ENVIRONMENTS.get(key, ENVIRONMENTS["kitchen"])
        self.stimuli = []
        self.time = 0.0
        for odor_key, x, y, z in self.environment.ambient_odors:
            self.add(odor_key, x, y, z)
        # Every environment gets one optimal-warmth patch so thermotaxis has a
        # positive target to compete with the hazards.
        self.add("warm_patch", -self.environment.size[0] * 0.3, 34.0, self.environment.size[2] * 0.25)

    def clear_stimuli(self, keep_sources: bool = True) -> None:
        if keep_sources:
            self.stimuli = [s for s in self.stimuli if s.key in ("warm_patch",) or
                            s.key in [k for k, _x, _y, _z in self.environment.ambient_odors]]
        else:
            self.stimuli = []

    def add(self, key: str, x: float, y: float, z: float) -> Stimulus | None:
        spec = CATALOG_BY_KEY.get(key)
        if spec is None:
            return None
        stim = Stimulus(
            id=f"s{self._next_id}",
            key=key,
            kind=spec.kind,
            x=x,
            y=y,
            z=z,
            params=dict(spec.params),
        )
        stim.last_x, stim.last_y, stim.last_z = x, y, z
        if spec.kind == "threat":
            # Threats oscillate about their drop point, so the drop point is the
            # centre of the arc rather than the live position.
            stim.params["home_x"] = x
            stim.params["home_y"] = y
            stim.params["home_z"] = z
        self._next_id += 1
        self.stimuli.append(stim)
        return stim

    def remove(self, stimulus_id: str) -> bool:
        before = len(self.stimuli)
        self.stimuli = [s for s in self.stimuli if s.id != stimulus_id]
        return len(self.stimuli) != before

    def move(self, stimulus_id: str, x: float, y: float, z: float) -> bool:
        for stim in self.stimuli:
            if stim.id == stimulus_id:
                stim.x, stim.y, stim.z = x, y, z
                if stim.kind == "odor":
                    # A moved source re-seeds its plume rather than leaving a trail
                    # behind, which keeps drag-and-drop responsive.
                    stim.puffs.clear()
                return True
        return False

    def set_param(self, stimulus_id: str, name: str, value: float) -> bool:
        for stim in self.stimuli:
            if stim.id == stimulus_id:
                stim.params[name] = value
                return True
        return False

    # -- fields -----------------------------------------------------------
    def wind_at(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Wind vector in mm/s: ambient drift plus the ceiling fan's flow field.

        A ceiling fan does not just blow down. Its downwash hits the surface and
        spreads outward as a **floor jet**, and that radial outflow is the thing
        that actually carries odour around a room. Modelling only the downwash put
        the plume straight through the counter and left the fly unable to smell
        food from anywhere, so the surface jet is not a flourish -- it is the
        transport that makes olfaction work at all.
        """
        wx, wy, wz = self.environment.wind
        for prop in self.environment.props:
            if prop.get("type") != "ceiling_fan":
                continue
            dx = x - prop["x"]
            dz = z - prop["z"]
            dy = y - prop["y"]
            radial = math.hypot(dx, dz)
            if radial < 1.0:
                radial = 1.0
            horizontal_falloff = 1.0 / (1.0 + (radial / (prop.get("r", 190.0) * 1.6)) ** 2)
            rpm = prop.get("rpm", 40.0)
            omega = rpm * 2.0 * math.pi / 60.0
            # Downwash: strongest under the blades, fading as it approaches the
            # surface because the flow has to turn outward.
            floor_y = self.terrain_height(x, z)
            height_above = max(0.0, y - floor_y)
            jet = clamp(1.0 - height_above / 90.0)
            push = 230.0 * horizontal_falloff * (1.0 - 0.85 * jet)
            wy += -push
            # Surface jet: radial outflow hugging the counter, with the swirl that
            # makes the plume wander rather than travel in a straight line.
            outflow = 190.0 * horizontal_falloff * jet
            swirl_phase = math.cos(omega * self.time * 0.12)
            wx += (dx / radial) * outflow * swirl_phase
            wz += (dz / radial) * outflow * swirl_phase
            swirl = 70.0 * horizontal_falloff * max(0.0, 1.0 - abs(dy) / 420.0)
            wx += -dz / radial * swirl * math.cos(omega * self.time)
            wz += dx / radial * swirl * math.cos(omega * self.time)
        return (wx, wy, wz)

    def odor_at(self, x: float, y: float, z: float, key: str | None = None) -> dict[str, float]:
        """Concentration per odour key at a point. Values are comparable, not calibrated."""
        out: dict[str, float] = {}
        for stim in self.stimuli:
            if stim.kind != "odor" or not stim.active:
                continue
            if key is not None and stim.key != key:
                continue
            sigma = stim.params.get("sigma", 40.0)
            strength = stim.params.get("strength", 1.0)
            dx = x - stim.x
            dy = y - stim.y
            dz = z - stim.z
            r2 = dx * dx + dy * dy + dz * dz
            # Near-field kernel: the source itself always smells.
            total = strength * math.exp(-r2 / (2.0 * sigma * sigma)) * 0.9
            # Room-scale accumulation. A continuously emitting source in an enclosed
            # space does not only exist as discrete filaments -- odour builds up and
            # a shallow gradient spans the whole room. Without this term a fan can
            # leave the fly standing in an odour hole with no usable global
            # gradient, which made chemotaxis fail outright rather than just being
            # hard.
            broad_sigma = stim.params.get("broad_sigma", 0.0)
            if broad_sigma > 0.0:
                total += (
                    strength
                    * stim.params.get("broad_strength", 0.3)
                    * math.exp(-r2 / (2.0 * broad_sigma * broad_sigma))
                )
            for puff in stim.puffs:
                dx = x - puff.x
                dy = y - puff.y
                dz = z - puff.z
                total += puff.strength * math.exp(
                    -(dx * dx + dy * dy + dz * dz) / (2.0 * puff.sigma * puff.sigma)
                )
            if total > 0.002:
                out[stim.key] = out.get(stim.key, 0.0) + total
        return out

    def odor_total(self, x: float, y: float, z: float) -> float:
        return sum(self.odor_at(x, y, z).values())

    def valence_at(self, x: float, y: float, z: float) -> tuple[float, float, dict[str, float]]:
        """Blend every odor present into ``(attractant, aversive, raw)``."""
        raw = self.odor_at(x, y, z)
        attract = 0.0
        aversive = 0.0
        for key, value in raw.items():
            spec = CATALOG_BY_KEY.get(key)
            if spec is None:
                continue
            valence = spec.params.get("valence", 0.0)
            if valence > 0:
                attract += value * valence
            else:
                aversive += value * -valence
        return attract, aversive, raw

    def temperature_at(self, x: float, y: float, z: float) -> float:
        temp = self.environment.ambient_temp_c
        for stim in self.stimuli:
            if stim.kind != "thermal" or not stim.active:
                continue
            sigma = stim.params.get("sigma", 80.0)
            # Steady-state diffusion from a small sphere falls off as 1/r, not as a
            # Gaussian. Halving the Gaussian at r=sigma matches that shape near the
            # source while staying bounded far away.
            dx = x - stim.x
            dy = y - stim.y
            dz = z - stim.z
            r2 = dx * dx + dy * dy + dz * dz
            falloff = 1.0 / (1.0 + r2 / (sigma * sigma))
            temp += stim.params.get("delta_c", 0.0) * falloff
        return temp

    def light_at(self, x: float, y: float, z: float) -> tuple[float, tuple[float, float, float]]:
        """Return ``(intensity, dominant direction toward the light)``."""
        intensity = self.environment.ambient_light
        best = 0.0
        direction = (0.0, 1.0, 0.0)
        for prop in self.environment.props:
            if prop.get("type") == "sun":
                intensity += 1.2
                best = max(best, 1.2)
                direction = (-700.0, 700.0, -500.0)
        for stim in self.stimuli:
            if stim.kind != "light" or not stim.active:
                continue
            dx = stim.x - x
            dy = stim.y - y
            dz = stim.z - z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            radius = stim.params.get("radius", 900.0)
            falloff = 1.0 / (1.0 + (distance / radius) ** 2)
            contribution = stim.params.get("intensity", 1.0) * falloff
            intensity += contribution
            if contribution > best:
                best = contribution
                direction = (dx / distance, dy / distance, dz / distance)
        return intensity, direction

    # -- threats ----------------------------------------------------------
    def threat_report(self, x: float, y: float, z: float) -> dict:
        """Looming and proximity as seen from the fly's position.

        Looming is the derivative of angular size, which is what a lobula giant
        movement detector actually computes. A large object moving slowly can
        produce less looming than a small one moving fast, and the model respects
        that instead of keying off distance.
        """
        nearest = None
        for stim in self.stimuli:
            if stim.kind != "threat" or not stim.active:
                continue
            dx = stim.x - x
            dy = stim.y - y
            dz = stim.z - z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            radius = stim.params.get("radius", 60.0)
            angular = 2.0 * math.atan2(radius, max(1.0, distance))
            # Closing speed in mm/s: the threat's velocity projected onto the line of
            # sight, positive when it is coming toward the fly. The fly's own motion
            # is included, which is correct -- optic flow is relative.
            closing = 0.0
            if distance > 0.001:
                closing = -(
                    stim.vel_x * dx + stim.vel_y * dy + stim.vel_z * dz
                ) / distance
            # d(theta)/dt for an approaching sphere, in radians per second.
            r2 = max(1.0, distance * distance)
            dtheta_dt = 2.0 * radius * closing / (r2 + radius * radius)
            stim.angular_size = angular
            stim.loom_rate = dtheta_dt
            danger = stim.params.get("danger", 1.0)
            # Looming dominates, but raw proximity counts too: a stationary hand
            # resting on top of the fly is a threat even without an approach.
            score = (
                max(0.0, dtheta_dt) * 1.1
                + clamp(1.0 - distance / 150.0) * 1.5 * danger
            )
            entry = {
                "id": stim.id,
                "key": stim.key,
                "distance": round(distance, 2),
                "angularSize": round(angular, 4),
                "loomRate": round(dtheta_dt, 4),
                "score": round(score * danger, 4),
                "position": [round(stim.x, 2), round(stim.y, 2), round(stim.z, 2)],
            }
            if nearest is None or entry["score"] > nearest["score"]:
                nearest = entry
        if nearest is None:
            return {"score": 0.0, "distance": 9999.0, "loomRate": 0.0, "angularSize": 0.0, "key": None}
        return nearest

    def terrain_height(self, x: float, z: float) -> float:
        """Height of the highest walkable surface under ``(x, z)``.

        Only the box-shaped props are walkable (counter, bench, ground, floor).
        Cylinders like a fruit bowl or a petri dish are treated as obstacles the fly
        walks around rather than surfaces it stands on, which is a deliberate
        simplification: a fly *does* stand on a petri dish lid, but modelling every
        prop's upper surface is not worth the collision code here.
        """
        height = 0.0
        for prop in self.environment.props:
            if "sx" not in prop or "sy" not in prop:
                continue
            if prop.get("type") == "wall":
                continue
            half_x = prop["sx"] * 0.5
            half_z = prop["sz"] * 0.5
            if abs(x - prop["x"]) <= half_x and abs(z - prop["z"]) <= half_z:
                top = prop["y"] + prop["sy"] * 0.5
                height = max(height, top)
        return height

    def hazard_at(self, x: float, y: float, z: float) -> float:
        """Thermal danger 0..1. Cold is uncomfortable; heat is lethal."""
        temp = self.temperature_at(x, y, z)
        if temp <= OPTIMAL_TEMP_C:
            return clamp((OPTIMAL_TEMP_C - temp) / 22.0) * 0.55
        return clamp((temp - OPTIMAL_TEMP_C) / (LETHAL_TEMP_C - OPTIMAL_TEMP_C))

    # -- integration ------------------------------------------------------
    def step(self, dt: float) -> None:
        self.time += dt
        total_puffs = 0

        for stim in self.stimuli:
            if not stim.active:
                continue
            if stim.kind == "odor":
                sigma0 = stim.params.get("sigma", 40.0)
                strength = stim.params.get("strength", 1.0)
                interval = stim.params.get("interval", 1.8)
                life = stim.params.get("life", 8.0)

                stim.emit_timer -= dt
                if stim.emit_timer <= 0.0 and len(stim.puffs) < self.MAX_PUFFS_PER_SOURCE:
                    stim.emit_timer = interval * self.random.uniform(0.7, 1.3)
                    local = self.wind_at(stim.x, stim.y, stim.z)
                    stim.puffs.append(
                        Puff(
                            x=stim.x + self.random.gauss(0.0, 6.0),
                            y=stim.y + self.random.gauss(0.0, 6.0),
                            z=stim.z + self.random.gauss(0.0, 6.0),
                            sigma=sigma0 * 0.55,
                            strength=strength * self.random.uniform(0.75, 1.05),
                            vx=local[0] * 0.7,
                            vy=local[1] * 0.35,
                            vz=local[2] * 0.7,
                        )
                    )

                alive: list[Puff] = []
                for puff in stim.puffs:
                    puff.age += dt
                    if puff.age > life:
                        continue
                    # Turbulent diffusion grows the cloud, and it steers toward the
                    # wind *at its own location* rather than carrying its launch
                    # velocity forever, so puffs ride the fan's floor jet outward.
                    puff.sigma += 14.0 * dt
                    local = self.wind_at(puff.x, puff.y, puff.z)
                    blend = min(1.0, 1.6 * dt)
                    puff.vx += (local[0] - puff.vx) * blend
                    puff.vy += (local[1] - puff.vy) * blend
                    puff.vz += (local[2] - puff.vz) * blend
                    puff.x += puff.vx * dt
                    puff.y += puff.vy * dt
                    puff.z += puff.vz * dt
                    # A plume cannot sink through a solid surface. Clamping to the
                    # terrain is what makes the odour pool on the countertop and
                    # flow along it, which is where a walking fly can smell it.
                    floor = self.terrain_height(puff.x, puff.z) + 3.0
                    if puff.y < floor:
                        puff.y = floor
                        puff.vy = 0.0
                    # The cloud disperses: strength decays while sigma grows, so a
                    # plume weakens as it spreads rather than staying concentrated.
                    puff.strength *= math.exp(-0.12 * dt)
                    alive.append(puff)
                stim.puffs = alive
                total_puffs += len(alive)

            elif stim.kind == "threat":
                self._advance_threat(stim, dt)

        # Global puff budget: keep the cheapest sources alive but bound the cost.
        if total_puffs > self.MAX_PUFFS_TOTAL:
            excess = total_puffs - self.MAX_PUFFS_TOTAL
            for stim in self.stimuli:
                if stim.kind == "odor" and stim.puffs and excess > 0:
                    drop = min(excess, len(stim.puffs))
                    stim.puffs = stim.puffs[drop:]
                    excess -= drop

    def _advance_threat(self, stim: Stimulus, dt: float) -> None:
        """Move a threat through an arc that genuinely closes on the fly.

        Two earlier versions of this failed for the same underlying reason, and both
        failures looked like "the escape reflex is broken" rather than like geometry:

        1. A swatter that swept **sideways at constant height** produces exactly zero
           looming, because the angular-size derivative of a tangential pass is zero.
        2. A swatter that swept sideways *while* descending only arrived at the fly's
           lateral position once it was already far below it.

        A real swat descends onto the target and follows it, so threats track the
        simulation's focus point and their arc is centred on it.
        """
        stim.last_x, stim.last_y, stim.last_z = stim.x, stim.y, stim.z
        stim.phase += dt
        speed = stim.params.get("speed", 800.0)
        step = max(1e-6, dt)
        swing = stim.params.get("swing", 260.0)

        if stim.params.get("track", True):
            self._track_focus(stim, dt)

        home_x = stim.params.get("home_x", stim.x)
        home_y = stim.params.get("home_y", stim.y)
        home_z = stim.params.get("home_z", stim.z)
        dip_scale = stim.params.get("dip", 0.35)

        # One period formula for every threat, derived from its own speed. The hand
        # used to have a hand-picked 6.5 s cycle, which made it descend at 71 mm/s --
        # slow enough that its looming rate never once crossed the escape threshold,
        # so a hand hovering over the fly produced no reaction at all.
        period = max(0.8, 4.0 * swing * dip_scale / max(1.0, speed))
        u = (stim.phase % period) / period
        dip = 0.5 - 0.5 * math.cos(u * 2.0 * math.pi)

        if stim.key == "hand":
            # Large, slow relative to its size, and descending almost straight down.
            stim.x = home_x + swing * 0.28 * math.sin(u * math.pi)
            stim.z = home_z + swing * 0.18 * math.sin(u * 3.0 * math.pi)
            stim.y = home_y - swing * dip_scale * dip
        elif stim.key == "swatter":
            stim.x = home_x + swing * 0.6 * math.sin(u * 2.0 * math.pi)
            stim.y = home_y - swing * dip_scale * dip
            stim.z = home_z
        else:
            # Erratic predator: an incommensurate pair, so the path never repeats.
            stim.x = home_x + swing * 0.55 * math.sin(u * 2.0 * math.pi)
            stim.z = home_z + swing * 0.45 * math.cos(u * 3.0 * math.pi)
            stim.y = home_y - swing * dip_scale * dip

        # Publish the velocity the looming detector needs, in mm/s.
        stim.vel_x = (stim.x - stim.last_x) / step
        stim.vel_y = (stim.y - stim.last_y) / step
        stim.vel_z = (stim.z - stim.last_z) / step

    def _track_focus(self, stim: Stimulus, dt: float) -> None:
        """Drift a threat's home point toward the focus, hovering above the surface."""
        track_speed = stim.params.get("track_speed", 42.0)
        home_x = stim.params.get("home_x", stim.x)
        home_y = stim.params.get("home_y", stim.y)
        home_z = stim.params.get("home_z", stim.z)
        fx, fy, fz = self.focus
        dx = fx - home_x
        dz = fz - home_z
        distance = math.hypot(dx, dz)
        if distance > 1.0:
            step = min(distance, track_speed * dt)
            home_x += dx / distance * step
            home_z += dz / distance * step
        # Hover. The vertical home point stays a fixed height above the fly, so the
        # arc always has something to converge on rather than bottoming out below it.
        hover = stim.params.get("hover", 55.0)
        target_y = fy + hover
        home_y += (target_y - home_y) * min(1.0, 1.2 * dt)
        home_y = min(home_y, self.environment.size[1] - 20.0)
        stim.params["home_x"] = home_x
        stim.params["home_y"] = home_y
        stim.params["home_z"] = home_z

    # -- serialisation ----------------------------------------------------
    def to_json(self) -> dict:
        return {
            "environment": self.environment.to_json(),
            "time": round(self.time, 3),
            "stimuli": [s.to_json() for s in self.stimuli],
        }
