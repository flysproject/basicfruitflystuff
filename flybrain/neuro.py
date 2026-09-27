"""Endocrine and internal-state model for the fly.

The neuromodulators here are not decoration: they are the coefficients of the
agent's own decision policy. Each one is modelled the way the biology works --
a **tonic** level the user sets with a slider, plus **phasic** transients the
simulation fires in response to events:

* Tonic dopamine sets how much the fly cares about a goal at all, and how fast it
  moves toward one.
* A phasic dopamine pulse fires on reward prediction, which is what a
  reward-prediction-error signal is. Feeding, finding food, or surviving a
  threat all produce one.
* Octopamine is the insect fight-or-flight axis. A looming shadow produces a
  spike with a short decay; high tonic octopamine lowers the threshold for
  takeoff and raises metabolic burn.
* Serotonin slows locomotion and raises the cost of switching behaviour, so a
  high-5-HT fly commits to whatever it is doing.
* Neuropeptide F is hunger drive. It does not move the fly so much as change what
  the fly is willing to risk: a starving fly enters zones a fed one avoids.
* DILPs set metabolic rate and satiety. High DILP makes energy burn faster and
  feeding less rewarding, which is the feedback loop that keeps NPF in check.
* The circadian clock runs a real phase, drives sleep pressure, and gates
  reactivity. At night with high sleep pressure the fly rests and startles less.

Tonic levels relax back to the slider value, so releasing a slider never leaves
the system in a permanent transient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["ModulatorSpec", "MODULATORS", "EndocrineSystem", "InternalState"]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else high if value > high else value


def approach(value: float, target: float, rate: float, dt: float) -> float:
    """Exponential relaxation that is stable for any dt (never overshoots)."""
    if rate <= 0.0:
        return target
    return target + (value - target) * math.exp(-rate * dt)


@dataclass(frozen=True)
class ModulatorSpec:
    key: str
    label: str
    short: str
    color: str
    blurb: str
    effect: str
    default: float
    # How fast a phasic transient decays back to tonic, in 1/seconds.
    decay: float = 0.55
    # How fast tonic tracks its slider target, in 1/seconds.
    relax: float = 0.35


MODULATORS: tuple[ModulatorSpec, ...] = (
    ModulatorSpec(
        key="DA",
        label="Dopamine",
        short="DA",
        color="#f9d423",
        blurb="Reward evaluation and goal pursuit. Dopaminergic neurons write valence onto "
        "mushroom-body output, so DA is the signal that makes an odour worth remembering.",
        effect="Raises walking and flight speed, extends goal persistence, and amplifies the "
        "reward pulse when food is reached. At 0 the fly still reacts to threats but stops "
        "chasing anything.",
        default=0.45,
    ),
    ModulatorSpec(
        key="OA",
        label="Octopamine",
        short="OA",
        color="#ff7a45",
        blurb="The insect analogue of noradrenaline. Drives flight readiness, arousal and "
        "metabolic hyperactivity.",
        effect="Above ~60% the fly takes off sooner. Expect roughly a 300% increase in wing-beat "
        "burst rate and a lower looming-detector threshold; energy drains proportionally faster.",
        default=0.25,
        decay=0.8,
    ),
    ModulatorSpec(
        key="5HT",
        label="Serotonin",
        short="5-HT",
        color="#7dd3fc",
        blurb="Behavioural inhibition, aggression and motor slowing. Opposes the arousal axis.",
        effect="Scales velocity down and raises the cost of switching behaviour, so the fly "
        "commits to its current action. Also widens the grooming window.",
        default=0.35,
    ),
    ModulatorSpec(
        key="NPF",
        label="Neuropeptide F",
        short="NPF",
        color="#a3e635",
        blurb="Hunger drive. NPF rises with energy deficit and promotes feeding and food search.",
        effect="Raises the risk the fly accepts: high NPF will walk into a hot zone or under a "
        "shadow for a sugar source, and drops the proboscis-extension threshold.",
        default=0.4,
        decay=0.3,
    ),
    ModulatorSpec(
        key="DILP",
        label="Insulin-like Peptides",
        short="DILP",
        color="#c084fc",
        blurb="Energy homeostasis. DILPs report nutrient status to the brain and set metabolic "
        "rate.",
        effect="Scales metabolic burn and satiety. High DILP means faster energy drain but a "
        "suppressed hunger drive; low DILP is the starvation-signalling state.",
        default=0.5,
        decay=0.3,
    ),
    ModulatorSpec(
        key="CLOCK",
        label="Circadian Clock",
        short="CLK",
        color="#5eead4",
        blurb="Period/timeless clock output. Gates rest, reactivity thresholds and odour "
        "sensitivity across the day.",
        effect="Below 0.30 the fly enters sleep-like rest with reduced startle and slow walking; "
        "above 0.70 it is a fully reactive daytime forager.",
        default=0.6,
        decay=0.2,
        relax=0.25,
    ),
)

MODULATOR_INDEX = {spec.key: i for i, spec in enumerate(MODULATORS)}


@dataclass
class InternalState:
    """Slow physiological variables the modulators act on."""

    # Starts mildly hungry so the fly forages within the first minute instead of
    # standing around grooming while the user waits for something to happen.
    energy: float = 0.60
    hunger: float = 0.35
    arousal: float = 0.35
    fear: float = 0.05
    sleep_pressure: float = 0.1
    reward_trace: float = 0.0
    punishment_trace: float = 0.0
    last_reward: float = 0.0
    time_of_day: float = 0.5
    sleep: bool = False

    def to_json(self) -> dict:
        return {
            "energy": round(self.energy, 4),
            "hunger": round(self.hunger, 4),
            "arousal": round(self.arousal, 4),
            "fear": round(self.fear, 4),
            "sleepPressure": round(self.sleep_pressure, 4),
            "rewardTrace": round(self.reward_trace, 4),
            "punishmentTrace": round(self.punishment_trace, 4),
            "timeOfDay": round(self.time_of_day, 4),
            "sleep": self.sleep,
        }


class EndocrineSystem:
    """Tonic/phasic modulator system plus the internal state it controls."""

    def __init__(self, day_length_s: float = 180.0) -> None:
        self.specs = MODULATORS
        self.tonic = [spec.default for spec in self.specs]
        self.target = [spec.default for spec in self.specs]
        self.phasic = [0.0 for _ in self.specs]
        self.state = InternalState()
        self.day_length_s = day_length_s
        # Transient bookkeeping for telemetry peaks.
        self.spikes: dict[str, float] = {spec.key: 0.0 for spec in self.specs}

    # -- slider interface -------------------------------------------------
    def set_slider(self, key: str, value: float) -> None:
        index = MODULATOR_INDEX.get(key)
        if index is None:
            return
        self.target[index] = clamp(value)

    def sliders(self) -> dict[str, float]:
        return {spec.key: round(self.tonic[i], 4) for i, spec in enumerate(self.specs)}

    def targets(self) -> dict[str, float]:
        return {spec.key: round(self.target[i], 4) for i, spec in enumerate(self.specs)}

    # -- phasic transients ------------------------------------------------
    def pulse(self, key: str, amount: float) -> None:
        """Fire a phasic transient on top of the tonic level.

        Pulses accumulate rather than overwrite, so a burst of events produces a
        larger transient than an isolated one -- which is how habituation and
        sensitisation fall out of the dynamics for free.
        """
        index = MODULATOR_INDEX.get(key)
        if index is None:
            return
        self.phasic[index] = min(1.4, self.phasic[index] + amount)
        self.spikes[key] = max(1.0, min(3.0, 1.0 + self.phasic[index]))

    def level(self, key: str) -> float:
        index = MODULATOR_INDEX.get(key)
        if index is None:
            return 0.0
        return clamp(self.tonic[index] + self.phasic[index])

    def raw(self, key: str) -> float:
        index = MODULATOR_INDEX.get(key)
        return 0.0 if index is None else self.tonic[index] + self.phasic[index]

    # -- integration ------------------------------------------------------
    def step(self, dt: float, activity_cost: float, feeding: float, threat: float) -> None:
        """Advance the modulator system by *dt* seconds.

        *activity_cost* is metabolic effort 0..1 for this tick, *feeding* the
        reward rate 0..1 (nectar uptake), *threat* the current threat proximity
        0..1. All three drive transients; nothing here reads the environment
        directly, which keeps the module testable on its own.
        """
        state = self.state

        for i, spec in enumerate(self.specs):
            if spec.key == "CLOCK":
                continue
            self.tonic[i] = approach(self.tonic[i], self.target[i], spec.relax, dt)
            self.phasic[i] = approach(self.phasic[i], 0.0, spec.decay, dt)

        # The clock advances on its own and is also draggable by the user.
        clock_index = MODULATOR_INDEX["CLOCK"]
        self.tonic[clock_index] = approach(
            self.tonic[clock_index], self.target[clock_index], self.specs[clock_index].relax, dt
        )
        self.phasic[clock_index] = approach(self.phasic[clock_index], 0.0, 0.2, dt)
        state.time_of_day = (state.time_of_day + dt / max(1.0, self.day_length_s)) % 1.0

        # -- physiology ---------------------------------------------------
        dilp = self.level("DILP")
        npf = self.level("NPF")
        oa = self.level("OA")
        da = self.level("DA")
        ht = self.level("5HT")
        clock = self.level("CLOCK")

        # Energy: burned by activity and by metabolic rate, restored by feeding.
        # The idle burn is set so an unfed fly starves over roughly ten minutes of
        # simulated time. A fly visibly fading in ninety seconds reads as a bug, not
        # as starvation, and hunger is supposed to be the slow variable here.
        burn = (0.0008 + 0.0045 * activity_cost) * (0.5 + dilp)
        state.energy = clamp(state.energy - burn * dt + feeding * 0.35 * dt)

        # Hunger is an energy deficit gated by insulin signalling.
        deficit = 1.0 - state.energy
        state.hunger = clamp(approach(state.hunger, clamp(0.75 * deficit + 0.35 * npf - 0.35 * dilp),
                                      1.6, dt))

        # Circadian gating: sleep pressure accumulates while awake and night
        # multiplies the sensitivity, which is what produces the rest cycle.
        night = 1.0 - clock
        state.sleep_pressure = clamp(
            state.sleep_pressure + (0.010 * activity_cost + 0.002) * dt
            - (0.02 if state.sleep else 0.0) * dt
        )
        state.sleep = state.sleep_pressure > 0.62 and night > 0.45

        # Arousal is octopamine minus serotonergic inhibition, gated by the clock.
        target_arousal = clamp(0.75 * oa + 0.45 * threat + 0.25 * state.fear + 0.35 * (clock - 0.4) - 0.45 * ht)
        if state.sleep:
            target_arousal *= 0.22
        state.arousal = approach(state.arousal, clamp(target_arousal), 2.2, dt)

        # Fear decays fast; it is a phasic quantity even though it feels sticky.
        state.fear = approach(state.fear, clamp(0.35 * threat), 1.4, dt)

        # Traces are the reward/punishment memory the learning rule reads.
        state.reward_trace = approach(state.reward_trace, clamp(0.4 * da + feeding * 0.6), 1.1, dt)
        state.punishment_trace = approach(state.punishment_trace, clamp(threat), 1.1, dt)
        state.last_reward = feeding

    # -- derived policy coefficients --------------------------------------
    def motor_gain(self) -> float:
        """Velocity multiplier from the modulator balance."""
        state = self.state
        gain = 0.45 + 0.85 * self.level("DA") + 0.9 * self.level("OA") - 0.55 * self.level("5HT")
        gain *= 1.0 + 0.35 * (state.energy - 0.5)
        gain *= 0.35 if state.sleep else 1.0
        return clamp(gain, 0.05, 2.4)

    def persistence(self) -> float:
        """How long a goal survives before the fly re-decides."""
        return clamp(0.25 + 0.75 * self.level("DA") + 0.35 * self.level("NPF"), 0.05, 1.5)

    def reactivity(self) -> float:
        """Threshold multiplier for threat response. Low means jumpy."""
        reaction = 0.35 + 0.9 * self.state.arousal + 0.5 * self.level("OA")
        reaction *= 1.0 - 0.55 * self.level("5HT") * 0.5
        reaction *= 0.3 if self.state.sleep else 1.0
        return clamp(reaction, 0.05, 2.0)

    def risk_tolerance(self) -> float:
        """How much threat the fly will walk into for food.

        Starvation is what makes a fly reckless, so this is dominated by NPF and
        the energy deficit rather than by dopamine.
        """
        return clamp(0.15 + 0.7 * self.state.hunger + 0.4 * self.level("NPF") - 0.25 * self.level("5HT"))

    def flight_readiness(self) -> float:
        return clamp(0.3 + 0.8 * self.level("OA") + 0.3 * self.state.arousal - 0.25 * self.level("5HT"))

    def feeding_drive(self) -> float:
        return clamp(0.15 + 0.9 * self.state.hunger + 0.35 * self.level("NPF") - 0.3 * self.level("DILP"))

    def learning_rate(self, mutation_gain: float = 1.0) -> float:
        """Plasticity rate, gated by arousal and the clock (sleep consolidates, but
        an asleep fly does not acquire)."""
        base = 0.10 + 0.25 * self.level("DA") + 0.12 * self.level("OA")
        return clamp(base * mutation_gain * (0.25 if self.state.sleep else 1.0), 0.0, 0.6)

    def to_json(self) -> dict:
        return {
            "levels": {spec.key: round(self.level(spec.key), 4) for spec in self.specs},
            "tonic": self.sliders(),
            "phasic": {spec.key: round(self.phasic[i], 4) for i, spec in enumerate(self.specs)},
            "spikes": {k: round(v, 3) for k, v in self.spikes.items()},
            "targets": self.targets(),
            "state": self.state.to_json(),
            "derived": {
                "motorGain": round(self.motor_gain(), 4),
                "persistence": round(self.persistence(), 4),
                "reactivity": round(self.reactivity(), 4),
                "riskTolerance": round(self.risk_tolerance(), 4),
                "flightReadiness": round(self.flight_readiness(), 4),
                "feedingDrive": round(self.feeding_drive(), 4),
                "learningRate": round(self.learning_rate(), 4),
            },
        }

    def apply_profile(self, payload: dict) -> None:
        """Restore an exported endocrine profile."""
        for key, value in (payload or {}).items():
            if key in MODULATOR_INDEX:
                self.set_slider(key, float(value))
        self.tonic = list(self.target)


def preset_profiles() -> dict[str, dict]:
    """The one-click experiment endpoints, as slider targets."""
    return {
        "starving-minefield": {"NPF": 0.95, "DILP": 0.1, "DA": 0.7, "OA": 0.6, "5HT": 0.05, "CLOCK": 0.7},
        "dopamine-overdrive": {"DA": 1.0, "OA": 0.7, "NPF": 0.6, "5HT": 0.05, "DILP": 0.4, "CLOCK": 0.8},
        "night-stupor": {"CLOCK": 0.05, "5HT": 0.8, "OA": 0.05, "DA": 0.2, "NPF": 0.35, "DILP": 0.6},
        "pavlovian": {"DA": 0.55, "OA": 0.35, "5HT": 0.3, "NPF": 0.55, "DILP": 0.45, "CLOCK": 0.65},
        "baseline": {spec.key: spec.default for spec in MODULATORS},
    }
