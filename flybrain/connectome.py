"""Connectome routing, associative memory and the mutation toggles.

Three things live here, because they are the same thing seen from different
angles:

1. :class:`CircuitSpec` -- a region-to-region connectivity graph. This is the
   interface real connectome data plugs into. Whole-brain FlyWire exports are
   ~127k neurons and millions of synapses, which is far too large to ship inside a
   desktop toy; so the contract is *pre-aggregated region circuits* in a small
   documented JSON shape, and :func:`aggregate_by_region` is provided to build one
   from a neuron table plus an edge list.
2. :class:`ConnectomeRouter` -- integrates activation over that graph with
   per-region time constants. Fast regions (optic lobe, giant fibre) let a looming
   shadow reach the escape command in a couple of ticks; slow regions (mushroom
   body) integrate evidence across seconds. That asymmetry is what makes learning
   and reflex read as different processes.
3. :class:`AssociativeMemory` -- mushroom-body plasticity as a Rescorla-Wagner
   rule over odour->valence weights. This is the classical conditioning the app
   exposes, and the mutation toggles act directly on this rule.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace

from .neuro import clamp

__all__ = [
    "CircuitEdge",
    "CircuitSpec",
    "DEFAULT_CIRCUIT",
    "ConnectomeSource",
    "BuiltinCircuitSource",
    "JsonCircuitSource",
    "aggregate_by_region",
    "AssociativeMemory",
    "ConnectomeRouter",
    "Mutations",
]


# --------------------------------------------------------------------------
# Circuit definition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitEdge:
    src: str
    dst: str
    weight: float
    kind: str = "excitatory"
    label: str = ""

    def to_json(self) -> dict:
        return {
            "src": self.src,
            "dst": self.dst,
            "weight": self.weight,
            "kind": self.kind,
            "label": self.label,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "CircuitEdge":
        return cls(
            src=payload["src"],
            dst=payload["dst"],
            weight=float(payload["weight"]),
            kind=payload.get("kind", "excitatory"),
            label=payload.get("label", ""),
        )


@dataclass
class CircuitSpec:
    """A named region-level circuit."""

    name: str
    edges: tuple[CircuitEdge, ...]
    # Per-region integration rate in 1/s. Fast reflex paths, slow memory.
    rates: dict[str, float] = field(default_factory=dict)
    source: str = "builtin"
    note: str = ""

    def incoming(self, region: str) -> tuple[CircuitEdge, ...]:
        return tuple(edge for edge in self.edges if edge.dst == region)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "note": self.note,
            "edges": [edge.to_json() for edge in self.edges],
            "rates": self.rates,
        }


# Hand-built circuit used when no connectome file is supplied. Every edge is a
# named pathway rather than a bare number, so the event log can say *why* a
# region lit up. Weights are qualitative, not measured.
DEFAULT_CIRCUIT = CircuitSpec(
    name="Drosophila region circuit (qualitative)",
    source="builtin",
    note=(
        "Hand-built region graph following the canonical olfactory, visual and "
        "escape pathways. Weights are qualitative. Supply a pre-aggregated FlyWire "
        "or Hemibrain circuit JSON to replace it."
    ),
    edges=(
        CircuitEdge("AL", "MB", 1.00, "excitatory", "projection neurons -> Kenyon cells"),
        CircuitEdge("AL", "LH", 0.80, "excitatory", "projection neurons -> lateral horn"),
        CircuitEdge("AL", "SEZ", 0.25, "excitatory", "odour-modulated feeding"),
        CircuitEdge("LH", "MB", 0.40, "excitatory", "innate valence into memory"),
        CircuitEdge("LH", "SEZ", -0.35, "inhibitory", "aversive feeding suppression"),
        CircuitEdge("LH", "CC", -0.30, "inhibitory", "avoidance brake on descent"),
        CircuitEdge("OL", "CX", 0.70, "excitatory", "motion parallax -> heading"),
        CircuitEdge("OL", "GF", 0.90, "excitatory", "lobula giant movement detector"),
        CircuitEdge("OL", "MB", 0.25, "excitatory", "visual context for memory"),
        CircuitEdge("GF", "CC", 1.00, "excitatory", "giant fibre -> tergotrochanteral motor"),
        CircuitEdge("GF", "CX", -0.40, "inhibitory", "escape overrides steering"),
        CircuitEdge("CX", "CC", 0.90, "excitatory", "descending steering command"),
        CircuitEdge("CX", "MB", 0.30, "excitatory", "recurrent state into memory"),
        CircuitEdge("MB", "CX", 0.50, "excitatory", "learned goal vector"),
        CircuitEdge("MB", "LH", -0.30, "inhibitory", "experience overrides instinct"),
        CircuitEdge("MB", "CC", 0.60, "excitatory", "learned motor bias"),
        CircuitEdge("SEZ", "CC", 0.50, "excitatory", "feeding motor programme"),
        CircuitEdge("SEZ", "MB", 0.30, "excitatory", "taste reinforces memory"),
        CircuitEdge("CC", "SEZ", 0.20, "excitatory", "corollary discharge to feeding"),
    ),
    rates={
        "OL": 16.0,  # vision is the fastest sense: escape latency starts here
        "GF": 22.0,
        "AL": 9.0,
        "SEZ": 8.0,
        "LH": 5.0,
        "CX": 4.0,
        "CC": 6.0,
        "MB": 1.6,  # memory integrates over seconds, not milliseconds
    },
)


# --------------------------------------------------------------------------
# Pluggable data sources
# --------------------------------------------------------------------------


class ConnectomeSource:
    """Base class for anything that can supply a :class:`CircuitSpec`."""

    name = "abstract"

    def available(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def load(self) -> CircuitSpec:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "available": self.available()}


class BuiltinCircuitSource(ConnectomeSource):
    name = "builtin"

    def available(self) -> bool:
        return True

    def load(self) -> CircuitSpec:
        return replace(DEFAULT_CIRCUIT)


class JsonCircuitSource(ConnectomeSource):
    """Load a pre-aggregated region circuit from JSON.

    Expected shape::

        {
          "name": "FlyWire region circuit",
          "note": "optional provenance string",
          "rates": {"OL": 16.0},
          "regions": ["AL", "MB", ...],          # optional; implied by edges
          "edges": [
            {"src": "AL", "dst": "MB", "weight": 0.83,
             "kind": "excitatory", "label": "PN -> KC"}
          ]
        }

    Region keys must match the atlas region keys (AL, OL, MB, CX, LH, SEZ, GF,
    CC). Unknown keys are kept but simply never light up, which makes a partial
    file safe to load.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.name = f"json:{os.path.basename(path)}"

    def available(self) -> bool:
        return os.path.isfile(self.path)

    def load(self) -> CircuitSpec:
        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        edges = tuple(CircuitEdge.from_json(item) for item in payload.get("edges", []))
        return CircuitSpec(
            name=payload.get("name", os.path.basename(self.path)),
            edges=edges,
            rates={k: float(v) for k, v in payload.get("rates", {}).items()},
            source=self.name,
            note=payload.get("note", ""),
        )


def aggregate_by_region(
    neurons: dict[str, dict],
    connections: list[dict],
    name: str = "aggregated",
    normalise: bool = True,
) -> CircuitSpec:
    """Turn a cell-level connectome export into a region-level circuit.

    This is the actual FlyWire/Hemibrain hook. Pass an adjacency list of
    ``{"pre": id, "post": id, "weight": synapse_count}`` plus a
    ``{id: {"region": "MB", ...}}`` annotation table, and get back the region
    graph the router consumes. Aggregating ~10^8 synapses is not something to do
    on the UI thread, so this is meant to run once, offline, and be cached as a
    :class:`JsonCircuitSource` file.

    With *normalise*, weights are rescaled to 0..1 across the strongest pathway,
    because the router only cares about relative drive.
    """
    totals: dict[tuple[str, str], float] = {}
    labels: dict[tuple[str, str], str] = {}
    for row in connections:
        pre = neurons.get(str(row.get("pre")))
        post = neurons.get(str(row.get("post")))
        if not pre or not post:
            continue
        src = pre.get("region")
        dst = post.get("region")
        if not src or not dst or src == dst:
            continue
        key = (str(src), str(dst))
        totals[key] = totals.get(key, 0.0) + float(row.get("weight", 1.0))
        if key not in labels:
            pre_type = pre.get("cellType") or pre.get("type") or "?"
            post_type = post.get("cellType") or post.get("type") or "?"
            labels[key] = f"{pre_type} -> {post_type}"

    if not totals:
        return replace(DEFAULT_CIRCUIT, name=name, source="aggregated(empty)")

    peak = max(totals.values()) if normalise else 1.0
    edges = tuple(
        CircuitEdge(src, dst, round(total / peak, 4), "excitatory", labels[(src, dst)])
        for (src, dst), total in sorted(totals.items(), key=lambda kv: -kv[1])
    )
    return CircuitSpec(
        name=name,
        edges=edges,
        rates=dict(DEFAULT_CIRCUIT.rates),
        source="aggregated",
        note=f"Aggregated from {len(connections)} connections across {len(neurons)} neurons.",
    )


# --------------------------------------------------------------------------
# Mutations
# --------------------------------------------------------------------------


@dataclass
class Mutations:
    """Genetic toggles. Each one is consulted by exactly the subsystem it should be.

    ``fruitless`` needs :class:`SocialDrive` in ``fly.py``; ``white`` multiplies
    visual gain in the router's drive table; ``dunce`` and ``rutabaga`` act on the
    mushroom-body learning rule only.
    """

    dunce: bool = False
    rutabaga: bool = False
    fruitless: bool = False
    white: bool = False

    @property
    def memory_retention(self) -> float:
        """Dunce blocks consolidation, so learned odour values wash out fast."""
        return 0.06 if self.dunce else 1.0

    @property
    def acquisition_gain(self) -> float:
        """Rutabaga lacks functional adenylyl cyclase: no acquisition at all."""
        return 0.0 if self.rutabaga else 1.0

    @property
    def visual_gain(self) -> float:
        """White mutants lose screening pigment, so contrast and gain drop."""
        return 0.45 if self.white else 1.0

    @property
    def social_gain(self) -> float:
        """Fruitless males court indiscriminately and aggregate abnormally."""
        return 0.25 if self.fruitless else 1.0

    def to_json(self) -> dict:
        return {
            "dunce": self.dunce,
            "rutabaga": self.rutabaga,
            "fruitless": self.fruitless,
            "white": self.white,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "Mutations":
        return cls(
            dunce=bool(payload.get("dunce", False)),
            rutabaga=bool(payload.get("rutabaga", False)),
            fruitless=bool(payload.get("fruitless", False)),
            white=bool(payload.get("white", False)),
        )


MUTATION_INFO = {
    "dunce": {
        "label": "dunce",
        "gene": "dnc",
        "blurb": "cAMP phosphodiesterase mutant. Learning happens but cannot consolidate: "
        "odour values decay within seconds.",
        "effect": "Memory retention set to 6%.",
    },
    "rutabaga": {
        "label": "rutabaga",
        "gene": "rut",
        "blurb": "Ca2+/calmodulin-dependent adenylyl cyclase mutant. The molecular coincidence "
        "detector is broken.",
        "effect": "Acquisition gain set to 0: the fly cannot learn at all.",
    },
    "fruitless": {
        "label": "fruitless",
        "gene": "fru",
        "blurb": "Sexual-behaviour circuit mutant that reshapes courtship and social spacing.",
        "effect": "Social gain set to 25%.",
    },
    "white": {
        "label": "white",
        "gene": "w",
        "blurb": "Screening-pigment transporter mutant. Eyes appear white and contrast "
        "sensitivity collapses.",
        "effect": "Visual gain set to 45%.",
    },
}


# --------------------------------------------------------------------------
# Mushroom body plasticity
# --------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    key: str
    print_name: str
    value: float = 0.0
    trials: int = 0
    last_us: float = 0.0

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "name": self.print_name,
            "value": round(self.value, 4),
            "trials": self.trials,
            "lastUs": round(self.last_us, 4),
        }


class AssociativeMemory:
    """Rescorla-Wagner valence learning over odour keys.

    ``V <- V + lr * (US - V)``, times the acquisition gain, with retention decay.
    That single rule reproduces the behaviours the UI exposes: acquisition,
    extinction (a US of 0 drives V back down), overshadowing (whichever odour is
    present gets the credit) and blocking (once V matches the US there is nothing
    left to learn).
    """

    def __init__(self, mutations: Mutations | None = None) -> None:
        self.mutations = mutations or Mutations()
        self.entries: dict[str, MemoryEntry] = {}
        self.trial_count = 0
        self.last_pair: tuple[str, float] | None = None

    def entry(self, key: str, print_name: str) -> MemoryEntry:
        found = self.entries.get(key)
        if found is None:
            found = MemoryEntry(key=key, print_name=print_name)
            self.entries[key] = found
        return found

    def value(self, key: str) -> float:
        found = self.entries.get(key)
        return found.value if found else 0.0

    def condition(
        self,
        key: str,
        print_name: str,
        us_valence: float,
        learning_rate: float,
        dt: float = 1.0,
    ) -> tuple[float, float]:
        """Apply one conditioning trial. Returns ``(before, after)`` for the log."""
        entry = self.entry(key, print_name)
        before = entry.value
        gain = self.mutations.acquisition_gain
        # Retention is a leak on the stored value, so dunce decays rather than
        # fails to store at all -- which is the actual mutant phenotype.
        leak = (1.0 - self.mutations.memory_retention) * 0.9
        delta = learning_rate * gain * (clamp(us_valence, -1.0, 1.0) - entry.value)
        entry.value = clamp(entry.value + delta * dt - entry.value * leak * dt, -1.0, 1.0)
        entry.trials += 1
        entry.last_us = us_valence
        self.trial_count += 1
        self.last_pair = (key, us_valence)
        return before, entry.value

    def decay(self, dt: float) -> None:
        """Spontaneous forgetting; the dunce mutation multiplies it hard."""
        leak = (1.0 - self.mutations.memory_retention) * 0.9 + 0.004
        for entry in self.entries.values():
            entry.value = clamp(entry.value * math.exp(-leak * dt), -1.0, 1.0)

    def best_positive(self) -> tuple[str, float] | None:
        candidates = [(e.print_name, e.value) for e in self.entries.values() if e.value > 0.05]
        return max(candidates, key=lambda kv: kv[1]) if candidates else None

    def best_negative(self) -> tuple[str, float] | None:
        candidates = [(e.print_name, e.value) for e in self.entries.values() if e.value < -0.05]
        return min(candidates, key=lambda kv: kv[1]) if candidates else None

    def to_json(self) -> dict:
        return {
            "entries": [e.to_json() for e in sorted(self.entries.values(), key=lambda e: -abs(e.value))],
            "trials": self.trial_count,
            "retention": self.mutations.memory_retention,
            "acquisitionGain": self.mutations.acquisition_gain,
        }


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


class ConnectomeRouter:
    """Integrate activation over the region graph, given sensory drive.

    Each region relaxes toward ``drive + sum(incoming weight * source activation)``
    at its own rate. Inputs saturate rather than summing without bound, so a region
    cannot exceed 1.0 and the graph cannot run away.
    """

    def __init__(
        self,
        region_keys: tuple[str, ...],
        circuit: CircuitSpec | None = None,
        mutations: Mutations | None = None,
    ) -> None:
        self.regions = tuple(region_keys)
        self.circuit = circuit or replace(DEFAULT_CIRCUIT)
        self.mutations = mutations or Mutations()
        self.activation: dict[str, float] = {key: 0.0 for key in self.regions}
        self.traffic: dict[str, float] = {}
        self._incoming: dict[str, list[tuple[int, CircuitEdge]]] = {}
        index = {key: i for i, key in enumerate(self.regions)}
        for key in self.regions:
            self._incoming[key] = [
                (index[edge.src], edge)
                for edge in self.circuit.incoming(key)
                if edge.src in index
            ]
        self.last_driven: dict[str, float] = {}

    def set_circuit(self, circuit: CircuitSpec) -> None:
        self.circuit = circuit
        index = {key: i for i, key in enumerate(self.regions)}
        for key in self.regions:
            self._incoming[key] = [
                (index[edge.src], edge)
                for edge in circuit.incoming(key)
                if edge.src in index
            ]

    def signal(self, key: str) -> float:
        return self.activation.get(key, 0.0)

    def step(self, dt: float, drive: dict[str, float], suppression: float = 0.0) -> None:
        """Advance activation by *dt*.

        *drive* injects sensory input per region. *suppression* is a global gain
        reduction (serotonin and sleep feed this), which is how behavioural
        inhibition slows the whole circuit rather than one pathway.
        """
        previous = dict(self.activation)
        global_gain = clamp(1.0 - 0.75 * suppression, 0.05, 1.0)
        traffic: dict[str, float] = {}

        for key in self.regions:
            incoming = 0.0
            for src_index, edge in self._incoming[key]:
                contribution = edge.weight * previous[self.regions[src_index]]
                incoming += contribution
                if abs(contribution) > 0.01:
                    traffic[f"{edge.src}->{edge.dst}"] = abs(contribution)
            target = drive.get(key, 0.0) + incoming
            # Saturating nonlinearity: prevents a strong pathway from dominating
            # without bound once several inputs converge on one region.
            target = math.tanh(max(0.0, target)) if target > 0 else -math.tanh(-target)
            rate = self.circuit.rates.get(key, 6.0)
            value = previous[key] + (target * global_gain - previous[key]) * min(1.0, rate * dt)
            self.activation[key] = clamp(value, -1.0, 1.0)

        self.traffic = dict(sorted(traffic.items(), key=lambda kv: -kv[1])[:8])
        self.last_driven = dict(drive)

    def strongest_pathway(self) -> tuple[str, float] | None:
        if not self.traffic:
            return None
        key = max(self.traffic, key=lambda k: self.traffic[k])
        return key, self.traffic[key]

    def readouts(self) -> dict[str, float]:
        """Named behavioural quantities derived from region activation."""
        return {
            "olfactorySalience": round(max(0.0, self.signal("AL")), 4),
            "visualLoom": round(max(0.0, self.signal("OL")), 4),
            "learnedValence": round(self.signal("MB"), 4),
            "headingDrive": round(self.signal("CX"), 4),
            "feedingMotor": round(max(0.0, self.signal("SEZ")), 4),
            "escapeDrive": round(max(0.0, self.signal("GF")), 4),
            "descendingDrive": round(max(0.0, self.signal("CC")), 4),
            "innateAversion": round(max(0.0, self.signal("LH")), 4),
        }

    def to_json(self) -> dict:
        return {
            "activation": {k: round(v, 4) for k, v in self.activation.items()},
            "traffic": {k: round(v, 4) for k, v in self.traffic.items()},
            "readouts": self.readouts(),
            "circuit": {"name": self.circuit.name, "source": self.circuit.source,
                        "note": self.circuit.note, "edges": len(self.circuit.edges)},
        }
