"""The fly agent: physics, motor output and the behaviour state machine.

Two design commitments matter here.

**Navigation is chemotactic, not telepathic.** The fly does not know where a food
source is. It samples odour at two antenna positions, steers toward whichever
antenna reads higher, and when it loses the plume entirely it *casts* --
oscillating its heading while holding upwind. That is klinotaxis, and it is why
the fly looks like it is searching rather than gliding straight to a treat. The
sensing is done against the same analytic field the renderer draws, so what you
see is what the fly smells.

**Second-order memory changes behaviour, it does not decorate it.** Learned odour
valence from the mushroom body is added to innate valence before the steering
decision is made, so a peppermint sniff that was once paired with a shock will
actually produce avoidance -- and a fly with the rutabaga mutation, whose
plasticity gain is zero, will not learn it in the first place.

Units are millimetres and seconds. Body length is 3.2 mm, which is real.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .connectome import AssociativeMemory, ConnectomeRouter, Mutations
from .env import CATALOG_BY_KEY, OPTIMAL_TEMP_C, World
from .neuro import EndocrineSystem, clamp

__all__ = ["Mode", "Fly", "SocialDrive", "MODE_INFO"]

BODY_LENGTH_MM = 3.2
WALK_STAND_HEIGHT = 1.1
WING_BEAT_HZ = 218.0


# Behavioural modes, in priority order: ESCAPE preempts everything below it.
MODES = ("SLEEP", "REST", "GROOM", "WALK", "FEED", "FLIGHT", "ESCAPE")

MODE_INFO = {
    "SLEEP": {"label": "Sleep rest", "color": "#5eead4", "blurb": "Circadian rest: low reactivity, no locomotion."},
    "REST": {"label": "Rest", "color": "#93c5fd", "blurb": "Standing still but alert; the fly re-decides each tick."},
    "GROOM": {"label": "Grooming", "color": "#c4b5fd", "blurb": "Antenna cleaning, leg brushing or wing wipe."},
    "WALK": {"label": "Walking", "color": "#4ade80", "blurb": "Tripod gait on the surface, chemically or thermally guided."},
    "FEED": {"label": "Feeding", "color": "#facc15", "blurb": "Proboscis extended, ingesting. Energy recovers."},
    "FLIGHT": {"label": "Flight", "color": "#38bdf8", "blurb": "Wing buzzing with steering banking."},
    "ESCAPE": {"label": "Escape", "color": "#f87171", "blurb": "Reflex takeoff, zigzag, or a full rotational flip."},
}

GROOM_VARIANTS = ("antenna", "front_legs", "wings", "abdomen")

# Minimum time in a mode before the decision circuit may leave it. Without this the
# thresholds fight each other and grooming and foraging flap back and forth every
# couple of seconds, which reads as a bug even though each decision is locally
# reasonable. Escape, feeding and sleep are exempt: those are the states that are
# supposed to preempt.
MIN_DWELL_S = {
    "REST": 0.9,
    "GROOM": 3.0,
    "WALK": 3.0,
    "FLIGHT": 1.2,
    "FEED": 1.6,
    "SLEEP": 4.0,
    "ESCAPE": 0.35,
}
PREEMPTIVE = ("ESCAPE", "FEED", "SLEEP")


@dataclass
class SocialDrive:
    """Minimal social coupling, used for the swarming mode.

    Real social behaviour is courtship, aggregation and aggression, all heavily
    shaped by fruitless. This keeps the same shape at a fraction of the size: flies
    are attracted to each other (aggregation), compete for food when close, and the
    fruitless mutation scrambles the attraction.
    """

    aggregation: float = 0.55
    competition: float = 0.7

    def influence(self, fly: "Fly", others: list["Fly"], mutations: Mutations) -> tuple[float, float]:
        """Return ``(attraction, competition)`` scaled by distance to neighbours."""
        if not others:
            return (0.0, 0.0)
        attraction = 0.0
        competition = 0.0
        for other in others:
            if other is fly:
                continue
            distance = math.dist(fly.position, other.position)
            attraction += self.aggregation * mutations.social_gain * math.exp(-distance / 90.0)
            if other.mode == "FEED":
                competition += self.competition * math.exp(-distance / 45.0)
        count = max(1, len(others) - 1)
        return (attraction / count, competition / count)


@dataclass
class Fly:
    id: str = "fly-0"
    name: str = "Fly 1"
    x: float = 0.0
    y: float = WALK_STAND_HEIGHT
    z: float = 0.0
    heading: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    speed: float = 0.0
    mode: str = "REST"
    mode_time: float = 0.0
    goal: str = "explore"
    goal_age: float = 0.0
    gait_phase: float = 0.0
    wing_phase: float = 0.0
    wing_amp: float = 0.0
    groom_variant: str = "antenna"
    groom_phase: float = 0.0
    proboscis: float = 0.0
    antenna_swipe: float = 0.0
    flip: float = 0.0
    casting: float = 0.0
    energy_spent: float = 0.0
    distance_travelled: float = 0.0
    flying_time: float = 0.0
    resting_time: float = 0.0
    feed_events: int = 0
    escape_events: int = 0
    grooming_events: int = 0
    last_threat_score: float = 0.0
    last_loom_rate: float = 0.0
    last_odor: dict[str, float] = field(default_factory=dict)
    last_attract: float = 0.0
    last_aversive: float = 0.0
    last_learned: float = 0.0
    last_temp_c: float = 24.0
    last_hazard: float = 0.0
    last_light: float = 0.0
    last_light_dir: tuple[float, float, float] = (0.0, 1.0, 0.0)
    attended_odor: tuple[str, str] | None = None
    wing_freq_hz: float = WING_BEAT_HZ
    plume_confidence: float = 0.0
    last_value_gradient: tuple[float, float] = (0.0, 0.0)
    # Refractory window after a takeoff. Escape pulses octopamine, which raises
    # reactivity and *lowers* the escape threshold, so without a refractory the
    # reflex re-triggers itself and the fly spends half its life mid-escape. Real
    # flies show both motor refractoriness after takeoff and habituation to
    # repeated identical looming.
    escape_refractory: float = 0.0
    # Grooming bookkeeping: how long the current bout lasts and when the next one
    # may begin. Both are re-rolled per bout, so the rhythm never becomes periodic.
    _groom_bout_s: float = 9.0
    _next_groom_at: float = 3.0
    footprint: list[tuple[float, float, float]] = field(default_factory=list)
    rng: random.Random = field(default_factory=lambda: random.Random(11))

    def __post_init__(self) -> None:
        if not isinstance(self.rng, random.Random):  # pragma: no cover - defensive
            self.rng = random.Random(11)

    # -- geometry ---------------------------------------------------------
    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def forward(self) -> tuple[float, float, float]:
        return (math.sin(self.heading), 0.0, math.cos(self.heading))

    def is_airborne(self) -> bool:
        return self.mode in ("FLIGHT", "ESCAPE")

    # -- main update ------------------------------------------------------
    def update(
        self,
        dt: float,
        world: World,
        neuro: EndocrineSystem,
        router: ConnectomeRouter,
        memory: AssociativeMemory,
        mutations: Mutations,
        social: SocialDrive | None = None,
        others: list["Fly"] | None = None,
        events: list | None = None,
    ) -> None:
        """Advance one fixed kernel step and emit any notable events."""
        log = events if events is not None else []
        self.mode_time += dt
        self.escape_refractory = max(0.0, self.escape_refractory - dt)

        self._sense(dt, world, memory, mutations)
        social_attract, competition = (
            social.influence(self, others or [], mutations) if social else (0.0, 0.0)
        )
        self._route(dt, neuro, router, social_attract)
        self._decide(dt, world, neuro, router, mutations, competition, log)
        self._drive(dt, world, neuro, router, memory, mutations, log)
        self._animate(dt, neuro)

    # -- sensing ----------------------------------------------------------
    def _sense(self, dt: float, world: World, memory: AssociativeMemory, mutations: Mutations) -> None:
        attract, aversive, raw = world.valence_at(self.x, self.y, self.z)
        self.last_odor = raw
        self.last_attract = attract
        self.last_aversive = aversive

        # Learned valence multiplies the strongest odour present, which is the
        # whole point of the mushroom body.
        learned = 0.0
        if raw:
            strongest = max(raw, key=lambda k: raw[k])
            spec = CATALOG_BY_KEY.get(strongest)
            learned = memory.value(strongest) * min(1.0, raw[strongest] * 2.0)
            self.attended_odor = (strongest, spec.label if spec else strongest)
        else:
            self.attended_odor = None
        self.last_learned = learned

        self.last_temp_c = world.temperature_at(self.x, self.y, self.z)
        self.last_hazard = world.hazard_at(self.x, self.y, self.z)
        light, self.last_light_dir = world.light_at(self.x, self.y, self.z)
        self.last_light = light * mutations.visual_gain

        report = world.threat_report(self.x, self.y, self.z)
        self.last_threat_score = report.get("score", 0.0)
        self.last_loom_rate = report.get("loomRate", 0.0)
        self._threat = report

        # Plume confidence is a slow memory of having been in odour. A plume is
        # intermittent by nature, so a fly that reset its intent on every missed
        # whiff would never commit to a search. Rising fast and decaying over about
        # six seconds is what turns a series of whiffs into one continuous bout.
        sensing = attract + aversive > 0.05
        rate = 3.0 if sensing else 0.17
        self.plume_confidence += ((1.0 if sensing else 0.0) - self.plume_confidence) * min(1.0, rate * dt)

        # Low oxygen / high CO2 is not modelled; the sleeping fly simply stops
        # taking in as much, which is what the reduced reactivity represents.

    def _route(self, dt: float, neuro: EndocrineSystem, router: ConnectomeRouter,
               social_attract: float) -> None:
        """Inject sensory drive into the connectome and integrate it.

        This is the bridge between physics and the brain visualiser: every region
        that lights up in the HUD is driven by a quantity measured here, not by a
        scripted animation.
        """
        drive = {
            "AL": min(1.0, 1.15 * (self.last_attract + self.last_aversive)),
            "OL": min(1.0, 0.55 * self.last_light + 1.25 * max(0.0, self.last_loom_rate) * 2.0),
            "SEZ": min(1.0, neuro.feeding_drive() * 0.6 + (0.5 if self.last_odor.get("sucrose") else 0.0)),
            "LH": min(1.0, 1.0 * self.last_aversive + 0.35 * social_attract),
            "GF": min(1.0, 1.6 * max(0.0, self.last_loom_rate) * 2.2),
            "CX": 0.18,
            "MB": 0.0,
            "CC": 0.0,
        }
        suppression = clamp(0.35 * neuro.level("5HT") if not neuro.state.sleep else 0.5)
        router.step(dt, drive, suppression=suppression)

    # -- decision ---------------------------------------------------------
    def _decide(
        self,
        dt: float,
        world: World,
        neuro: EndocrineSystem,
        router: ConnectomeRouter,
        mutations: Mutations,
        competition: float,
        log: list,
    ) -> None:
        reactions = neuro.reactivity()
        threat = self.last_threat_score
        hazard = self.last_hazard

        # --- reflex path: escape ------------------------------------------
        # The giant fibre does not wait for the decision circuit, so this check
        # runs before anything else and can preempt any mode.
        escape_threshold = 0.65 / max(0.2, reactions)
        if (
            self.escape_refractory <= 0.0
            and (threat > escape_threshold or hazard > 0.92)
            and self.mode != "ESCAPE"
        ):
            panic = threat > escape_threshold
            self._take_escape(world, neuro, router, panic, log)
            return

        # --- dwell-time gate ----------------------------------------------
        # Stay committed to the current behaviour unless something preemptive has
        # already fired above, or the fly has served its minimum time here.
        if self.mode_time < MIN_DWELL_S.get(self.mode, 0.5):
            return

        # --- sleep --------------------------------------------------------
        if neuro.state.sleep and self.mode not in ("ESCAPE",):
            if self.mode != "SLEEP":
                self._enter("SLEEP", log, f"{self.name} enters circadian rest")
            self.speed = 0.0
            return

        # --- feeding -------------------------------------------------------
        feeding_drive = neuro.feeding_drive()
        on_food = self._on_food_source(world)
        if on_food and feeding_drive > 0.35 and self.last_aversive < 0.55:
            if self.mode != "FEED":
                self._enter("FEED", log, f"{self.name} extends proboscis on {on_food[1]}")
                neuro.pulse("DA", 0.25)
            return

        # --- flight --------------------------------------------------------
        wants_height = self.last_hazard > 0.35 or (self.last_attract < 0.02 and self.mode == "FLIGHT")
        can_fly = neuro.flight_readiness() > 0.45 + 0.25 * (1.0 - reactions)
        if can_fly and (wants_height or self._needs_flight(world, neuro)):
            if self.mode != "FLIGHT":
                self._enter("FLIGHT", log, f"{self.name} takes off ({neuro.level('OA'):.0%} octopamine)")
                self.y = max(self.y, WALK_STAND_HEIGHT + 30.0)
            self.goal = "navigate"
            return

        # --- grooming ------------------------------------------------------
        # Grooming occupies the calm middle: aroused enough to be awake, but with
        # nothing pressing. Serotonin widens the window.
        # --- walk / rest ---------------------------------------------------
        # Motivation has to be *actionable*. Being two degrees off the thermal
        # optimum in a still room is not a reason to abandon grooming, so the
        # thermal term needs a real error or a real gradient to count.
        thermal_error = abs(self.last_temp_c - OPTIMAL_TEMP_C)
        thermal_pull = self._thermal_gradient(world)
        # Foraging costs energy, so it is gated on hunger: a fly that has just fed
        # stops chasing odour and grooms instead. Without this gate the room-scale
        # odour background kept the fly permanently motivated, which cost the whole
        # grooming and rest repertoire and made it look like a wind-up toy.
        hungry_enough = neuro.state.hunger > 0.22 or neuro.level("NPF") > 0.72
        food_motivation = self.plume_confidence > 0.14 and hungry_enough
        motivated = (
            food_motivation
            or self.last_aversive > 0.07
            or self.last_learned < -0.08
            or competition > 0.2
            or thermal_error > 3.0
            or (thermal_pull > 0.35 and self.mode != "WALK")
        )
        if motivated:
            if self.mode != "WALK":
                reason = self._motivation_reason(competition, thermal_error)
                self._enter("WALK", log, f"{self.name} begins foraging: {reason}")
            return

        # --- grooming ------------------------------------------------------
        # Grooming occupies the calm middle: awake, stationary, nothing pressing.
        # Serotonin widens the window, which is the behavioural-inhibition story.
        calm = (
            threat < 0.18
            and hazard < 0.2
            and self.last_aversive < 0.25
            and self.speed < 8.0
            and neuro.level("OA") < 0.60 + 0.3 * neuro.level("5HT")
        )
        if self.mode == "GROOM":
            if not calm:
                self._enter("REST", log, f"{self.name} stops grooming: {self._interrupt_reason(threat, hazard)}")
                return
            # Grooming is a bout with an end, not a terminal state. Without a
            # duration limit the fly groomed for 82% of a ten-minute run, which is
            # not a repertoire, it is a stuck state.
            if self.mode_time > self._groom_bout_s:
                self._groom_bout_s = self.rng.uniform(6.0, 16.0)
                self._next_groom_at = self.mode_time + self.rng.uniform(4.0, 12.0)
                self._enter("REST", log, f"{self.name} finishes grooming")
            return
        groom_bias = 0.45 + 0.5 * neuro.level("5HT")
        if calm and self.mode_time > self._next_groom_at and self.rng.random() < 0.6 * groom_bias * dt:
            self._groom_bout_s = self.rng.uniform(6.0, 16.0)
            self.groom_variant = self.rng.choice(GROOM_VARIANTS)
            self._enter("GROOM", log, f"{self.name} starts {self.groom_variant} grooming")
            self.grooming_events += 1
            return

        if self.mode not in ("REST", "GROOM"):
            self._enter("REST", log, f"{self.name} settles")

    def _motivation_reason(self, competition: float, thermal_error: float) -> str:
        if self.last_aversive > 0.07:
            return "avoiding an aversive odour"
        if self.last_learned > 0.05:
            return "acting on a learned odour value"
        if self.last_attract > 0.05:
            return "following an odour plume"
        if competition > 0.2:
            return "contesting a food source"
        if thermal_error > 3.0:
            return f"thermotaxis ({self.last_temp_c:.1f} C is outside preference)"
        return "exploration"

    def _interrupt_reason(self, threat: float, hazard: float) -> str:
        if threat > 0.18:
            return "threat nearby"
        if hazard > 0.2:
            return "thermal hazard"
        if self.last_aversive > 0.25:
            return "aversive odour"
        return "arousal"

    def _thermal_gradient(self, world: World) -> float:
        """How much better the temperature gets a short step to the left or right.

        This is the difference between *being* off-optimum and being able to *do*
        something about it, and only the latter should pull a fly off a grooming
        bout.
        """
        left = self._sample_temp(world, 9.0, -math.pi / 2.0)
        right = self._sample_temp(world, 9.0, math.pi / 2.0)
        best = min(abs(left - OPTIMAL_TEMP_C), abs(right - OPTIMAL_TEMP_C))
        return abs(self.last_temp_c - OPTIMAL_TEMP_C) - best

    def _sample_temp(self, world: World, distance: float, angle: float) -> float:
        offset = self.heading + angle
        return world.temperature_at(
            self.x + math.sin(offset) * distance, self.y, self.z + math.cos(offset) * distance
        )

    def _needs_flight(self, world: World, neuro: EndocrineSystem) -> bool:
        """Fly when the goal is far and the plume is broken up."""
        if self.mode == "FLIGHT":
            return True
        if self.last_attract > 0.5 or self.last_aversive > 0.5:
            return False
        octopamine = neuro.level("OA")
        return (
            octopamine > 0.62
            and self.mode_time > 0.5
            and self.rng.random() < 0.02 + 0.09 * octopamine
        )

    def _on_food_source(self, world: World) -> tuple[str, str] | None:
        """Is the fly standing on something it can actually eat?

        Contact matters: a fly finds sugar by tarsal taste, not by smell, so this
        requires proximity, not just a strong plume.
        """
        for stim in world.stimuli:
            if stim.kind != "odor":
                continue
            spec = CATALOG_BY_KEY.get(stim.key)
            if spec is None or spec.params.get("valence", 0.0) <= 0.4:
                continue
            # Contact is horizontal, and the vertical tolerance is generous: a fly
            # tastes with its tarsi and will settle on a surface a few tens of
            # millimetres above the one it was walking on. Requiring exact 3-D
            # proximity made a fly that had found the food still unable to eat it.
            horizontal = math.dist((self.x, self.z), (stim.x, stim.z))
            if horizontal < 24.0 and abs(self.y - stim.y) < 45.0:
                return (stim.key, spec.label)
        return None

    # -- motor output -----------------------------------------------------
    def _drive(
        self,
        dt: float,
        world: World,
        neuro: EndocrineSystem,
        router: ConnectomeRouter,
        memory: AssociativeMemory,
        mutations: Mutations,
        log: list,
    ) -> None:
        gain = neuro.motor_gain()
        if self.mode == "SLEEP":
            self.speed = 0.0
        elif self.mode == "ESCAPE":
            self._drive_escape(dt, world)
        elif self.mode == "FLIGHT":
            self._drive_flight(dt, world, neuro, router, gain)
        elif self.mode in ("WALK", "FEED"):
            self._drive_walk(dt, world, neuro, router, memory, gain, mutations)
        else:
            self.speed *= math.exp(-6.0 * dt)
            self.heading += self.rng.gauss(0.0, 0.35) * dt

        self._integrate(dt, world)

    def _drive_walk(
        self,
        dt: float,
        world: World,
        neuro: EndocrineSystem,
        router: ConnectomeRouter,
        memory: AssociativeMemory,
        gain: float,
        mutations: Mutations,
    ) -> None:
        target_speed = 22.0 * gain if self.mode == "WALK" else 4.0
        if neuro.state.sleep:
            target_speed *= 0.2
        self.speed += (target_speed - self.speed) * min(1.0, 4.0 * dt)

        # The baseline has to be wide enough to resolve a smooth, fan-circulated
        # plume. Six millimetres is a realistic antenna span and a useless gradient
        # estimator; twenty-five resolves the field the fly actually lives in.
        turn = self._steer(world, neuro, router, memory, mutations, probe=25.0)
        max_turn = math.radians(220.0) * gain * dt
        self.heading += clamp(turn, -max_turn, max_turn)
        self.gait_phase += dt * max(1.5, self.speed * 0.55)

    def _steer(
        self,
        world: World,
        neuro: EndocrineSystem,
        router: ConnectomeRouter,
        memory: AssociativeMemory,
        mutations: Mutations,
        probe: float,
    ) -> float:
        """Gradient-ascent steering, in radians per second.

        An earlier version compared two antenna samples a body length apart. That
        is the right *idea* and the wrong instrument here: a fan-circulated plume is
        smooth over tens of millimetres, so the two readings were nearly identical
        and the fly walked straight through its own food.

        What works is to finite-difference the **motivational value** field over a
        baseline wide enough to resolve the gradient, then steer along the
        direction of that gradient. Only the direction is trusted -- a dilute plume
        far from its source still defines a perfectly good heading, and using the
        direction rather than the magnitude is what lets the fly commit.
        """
        gradient = self._value_gradient(world, memory, neuro, probe)
        self.last_value_gradient = gradient
        gx, gz = gradient
        magnitude = math.hypot(gx, gz)
        heading_bias = router.signal("CX") * 0.25
        steer = heading_bias

        if magnitude > 1e-9 and self.plume_confidence > 0.05:
            desired = math.atan2(gx, gz)
            error = ((desired - self.heading + math.pi) % (2.0 * math.pi)) - math.pi
            # Saturate the gain so a steep gradient near the source does not spin
            # the fly at high speed.
            steer += clamp(error, -0.8, 0.8) * 3.2
            self.casting = max(0.0, self.casting - 2.5 * 0.016)
        else:
            # Nothing to climb: cast crosswind in an oscillating search, holding
            # roughly upwind so the next filament is intercepted.
            wind = world.wind_at(self.x, self.y, self.z)
            if abs(wind[0]) + abs(wind[2]) > 1.0:
                upwind = math.atan2(-wind[0], -wind[2])
                error = ((upwind - self.heading + math.pi) % (2.0 * math.pi)) - math.pi
                steer += clamp(error, -0.9, 0.9) * 2.2
            self.casting = min(1.0, self.casting + 1.5 * 0.016)
            steer += math.sin(self.mode_time * 4.2) * 1.9
        return steer

    def _value_gradient(
        self, world: World, memory: AssociativeMemory, neuro: EndocrineSystem, radius: float
    ) -> tuple[float, float]:
        """Central-difference gradient of net motivational value, in per-mm units.

        Value combines innate attraction, innate aversion weighted by how much risk
        the fly can currently afford, and the learned odour value the mushroom body
        has written. That is why a conditioned fly turns away from a peppermint
        plume it would otherwise have ignored.
        """
        aversion_weight = 1.0 + 1.4 * (1.0 - neuro.risk_tolerance())
        xp = self._sample_value(world, memory, self.x + radius, self.z, aversion_weight)
        xm = self._sample_value(world, memory, self.x - radius, self.z, aversion_weight)
        zp = self._sample_value(world, memory, self.x, self.z + radius, aversion_weight)
        zm = self._sample_value(world, memory, self.x, self.z - radius, aversion_weight)
        span = 2.0 * radius
        return ((xp - xm) / span, (zp - zm) / span)

    def _sample_value(
        self, world: World, memory: AssociativeMemory, x: float, z: float, aversion_weight: float
    ) -> float:
        """Net motivational value of standing at ``(x, z)``."""
        raw = world.odor_at(x, self.y, z)
        value = 0.0
        for key, concentration in raw.items():
            spec = CATALOG_BY_KEY.get(key)
            innate = spec.params.get("valence", 0.0) if spec else 0.0
            # Learned valence is accumulated per odour, so a shock-paired peppermint
            # carries negative value even though its innate valence is exactly zero.
            value += (innate * (1.0 if innate > 0 else aversion_weight) + memory.value(key)) * concentration
        temp = world.temperature_at(x, self.y, z)
        return value - abs(temp - OPTIMAL_TEMP_C) * 0.012

    def _drive_flight(
        self,
        dt: float,
        world: World,
        neuro: EndocrineSystem,
        router: ConnectomeRouter,
        gain: float,
    ) -> None:
        cruise = 380.0 * gain
        self.speed += (cruise - self.speed) * min(1.0, 2.4 * dt)
        # Bank into turns: roll leads the yaw, which is what makes a turn read as
        # a turn rather than a swivel.
        wind = world.wind_at(self.x, self.y, self.z)
        self.heading += math.sin(self.mode_time * 1.7) * 0.5 * dt * gain
        self.roll = clamp(-self.roll * 0.9 + self.speed * 0.0006 * math.sin(self.mode_time * 1.7), -0.6, 0.6)
        self.pitch = clamp(wind[1] * -0.0004 + (self.y - 120.0) * 0.0004, -0.5, 0.5)
        if self.y < 12.0:
            self.y = 12.0

    def _drive_escape(self, dt: float, world: World) -> None:
        if self.mode_time > 0.05 and self.flip < 1.0 and self.mode_time < 0.5:
            # Rotational flip during the first half second of the escape.
            self.flip = min(1.0, self.mode_time / 0.45)
        self.speed += (820.0 - self.speed) * min(1.0, 8.0 * dt)
        if self.mode_time < 0.25:
            self.y += 620.0 * dt
        # Evasive zigzag: an incommensurate pair, so the path is never repeatable.
        self.heading += math.sin(self.mode_time * 27.0) * 6.5 * dt + math.sin(self.mode_time * 41.0) * 2.4 * dt
        self.roll = clamp(math.sin(self.mode_time * 21.0) * 0.9, -1.0, 1.0)
        if self.mode_time > 0.75 + 0.4 * self.rng.random():
            self.flip = 0.0

    def _integrate(self, dt: float, world: World) -> None:
        if not self.is_airborne():
            self.y = max(WALK_STAND_HEIGHT, world.terrain_height(self.x, self.z) + WALK_STAND_HEIGHT)
        else:
            self.y += (self.pitch * 90.0) * dt
            self.y = max(4.0, self.y)
        step = self.speed * dt
        self.x += math.sin(self.heading) * step
        self.z += math.cos(self.heading) * step
        self.distance_travelled += step
        self.energy_spent += step * (0.010 if self.is_airborne() else 0.004)

        # Arena walls: a fly does not walk through a wall, it climbs or turns away.
        half = (world.environment.size[0] * 0.5, world.environment.size[1], world.environment.size[2] * 0.5)
        if self.x < -half[0]:
            self.x = -half[0]
            self.heading = math.pi - self.heading
        elif self.x > half[0]:
            self.x = half[0]
            self.heading = math.pi - self.heading
        if self.z < -half[2]:
            self.z = -half[2]
            self.heading = -self.heading
        elif self.z > half[2]:
            self.z = half[2]
            self.heading = -self.heading
        if self.y > half[1]:
            self.y = half[1]

        if self.is_airborne():
            self.flying_time += dt
        else:
            self.resting_time += dt

        if not self.footprint or math.dist(self.footprint[-1], self.position) > 6.0:
            self.footprint.append(self.position)
            if len(self.footprint) > 90:
                del self.footprint[0]

    # -- animation --------------------------------------------------------
    def _animate(self, dt: float, neuro: EndocrineSystem) -> None:
        """Advance every visible part so the renderer is a pure function of state."""
        flying = self.is_airborne()
        # Wing beat is real: 218 Hz base, raised by octopamine and by flight speed.
        self.wing_freq_hz = WING_BEAT_HZ * (1.0 + 0.28 * neuro.level("OA") + 0.0004 * self.speed)
        amp_target = 1.0 if flying else 0.0
        self.wing_amp += (amp_target - self.wing_amp) * min(1.0, 14.0 * dt)
        # Phase advances at the true rate. At 60 fps the renderer cannot show
        # individual strokes, so it strobes -- which is exactly what a real fly
        # looks like on camera.
        self.wing_phase = (self.wing_phase + self.wing_freq_hz * dt) % 1.0

        if self.mode == "GROOM":
            self.groom_phase = (self.groom_phase + dt * 3.4) % 1.0
            self.antenna_swipe = (
                0.5 + 0.5 * math.sin(self.groom_phase * 2.0 * math.pi)
                if self.groom_variant in ("antenna", "front_legs")
                else 0.0
            )
        else:
            self.antenna_swipe += (0.0 - self.antenna_swipe) * min(1.0, 6.0 * dt)
            self.groom_phase = 0.0

        if self.mode == "FEED":
            self.proboscis = min(1.0, self.proboscis + 3.5 * dt)
        else:
            self.proboscis = max(0.0, self.proboscis - 2.5 * dt)

        if not flying:
            self.gait_phase += dt * max(1.0, self.speed * 0.55)
        # Roll and pitch are owned by the mode drivers; only unwind them when the
        # fly is on the ground, so a banked flight turn does not flatten mid-turn.
        if not flying:
            self.roll += (0.0 - self.roll) * min(1.0, 3.0 * dt)
            self.pitch += (0.0 - self.pitch) * min(1.0, 3.0 * dt)
        if self.mode == "ESCAPE" and self.mode_time > 1.0:
            self.flip = max(0.0, self.flip - 2.0 * dt)

    # -- transitions ------------------------------------------------------
    def _enter(self, mode: str, log: list, message: str) -> None:
        if mode == self.mode:
            return
        previous = self.mode
        self.mode = mode
        self.mode_time = 0.0
        log.append(
            {
                "kind": "mode",
                "from": previous,
                "to": mode,
                "fly": self.id,
                "message": message,
            }
        )

    def _take_escape(
        self,
        world: World,
        neuro: EndocrineSystem,
        router: ConnectomeRouter,
        panic: bool,
        log: list,
    ) -> None:
        threat = self._threat
        # The reflex is a chain, and the log reports the chain rather than the
        # outcome: detector -> giant fibre -> octopamine -> takeoff.
        detail = (
            f"looming shadow detected (dtheta/dt={self.last_loom_rate:+.2f} rad/s) "
            f"-> Giant Fibre activated (GF={router.signal('GF'):.2f}) "
            f"-> octopamine spike -> escape takeoff"
            if panic
            else f"thermal hazard {self.last_hazard:.0%} exceeded tolerance -> escape takeoff"
        )
        self.escape_events += 1
        self.escape_refractory = 3.0 + 1.5 * self.rng.random()
        self.flip = 0.0
        neuro.pulse("OA", 0.55 if panic else 0.4)
        neuro.pulse("5HT", -0.15)
        neuro.pulse("DA", -0.1)
        neuro.state.fear = min(1.0, neuro.state.fear + (0.5 if panic else 0.35))
        # Launch away from the threat, not randomly.
        source = threat.get("position")
        if source and threat.get("score", 0.0) > 0.01:
            away = math.atan2(self.x - source[0], self.z - source[2])
            self.heading = away
        else:
            self.heading = self.rng.uniform(-math.pi, math.pi)
        self.speed = max(self.speed, 300.0)
        self._enter("ESCAPE", log, f"{self.name} ESCAPE: {detail}")
        log.append(
            {
                "kind": "escape",
                "fly": self.id,
                "severity": "alert",
                "message": detail,
            }
        )

    # -- serialisation ----------------------------------------------------
    def to_json(self, full: bool = True) -> dict:
        payload = {
            "id": self.id,
            "name": self.name,
            "position": [round(self.x, 2), round(self.y, 2), round(self.z, 2)],
            "heading": round(self.heading, 4),
            "pitch": round(self.pitch, 4),
            "roll": round(self.roll, 4),
            "speed": round(self.speed, 2),
            "mode": self.mode,
            "modeTime": round(self.mode_time, 2),
            "goal": self.goal,
            "gaitPhase": round(self.gait_phase, 3),
            "wingPhase": round(self.wing_phase, 3),
            "wingAmp": round(self.wing_amp, 3),
            "wingHz": round(self.wing_freq_hz, 1),
            "proboscis": round(self.proboscis, 3),
            "antennaSwipe": round(self.antenna_swipe, 3),
            "groomVariant": self.groom_variant,
            "flip": round(self.flip, 3),
            "casting": round(self.casting, 3),
            "senses": {
                "attract": round(self.last_attract, 4),
                "aversive": round(self.last_aversive, 4),
                "learned": round(self.last_learned, 4),
                "tempC": round(self.last_temp_c, 2),
                "hazard": round(self.last_hazard, 4),
                "light": round(self.last_light, 3),
                "loomRate": round(self.last_loom_rate, 4),
                "threat": round(self.last_threat_score, 4),
            },
            "stats": {
                "distance": round(self.distance_travelled, 1),
                "flyingTime": round(self.flying_time, 1),
                "restingTime": round(self.resting_time, 1),
                "escapes": self.escape_events,
                "groomingEvents": self.grooming_events,
            },
        }
        if full:
            payload["odor"] = {k: round(v, 4) for k, v in self.last_odor.items()}
        return payload


# ``_probe`` needs the memory at hand; passing the object through every sensor call
# would be noise, so the active memory is bound per update.
