#!/usr/bin/env python3
"""Reproduce the behavioural claims made in README.md and report pass or fail.

The README asserts specific things about how the fly behaves — a baseline mode mix, a
count of escape responses per threat, the learned value of a conditioned odour, and a
performance budget. Those numbers are only worth publishing if anyone can regenerate
them, so this runs exactly the scenarios and checks the results against the documented
ranges.

    python verify.py             # everything (about four minutes)
    python verify.py --quick     # shorter baseline, no performance sweep

Exit code is 0 only if every check passes, so it doubles as a CI gate.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flybrain.atlas import load_or_build  # noqa: E402
from flybrain.kernel import Kernel  # noqa: E402

STEP = 1.0 / 120.0
SAMPLE_EVERY = 40  # the sampling cadence the headless report uses

# Where the README says a threat is dropped. High and off-centre, so the fly has to be
# approached by the looming stimulus rather than starting inside it.
THREAT_AT = (140.0, 80.0, 140.0)

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str) -> bool:
    results.append((bool(ok), label, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<44} {detail}")
    return bool(ok)


def build(atlas, **kwargs) -> Kernel:
    return Kernel(atlas, root=ROOT, **kwargs)


def run(kernel: Kernel, seconds: float, sample: bool = False) -> tuple[dict, float]:
    """Step the kernel for *seconds* of simulated time. Returns (mode counts, wall)."""
    modes: collections.Counter = collections.Counter()
    started = time.perf_counter()
    steps = int(round(seconds / STEP))
    for index in range(steps):
        kernel._step(STEP)
        if sample and index % SAMPLE_EVERY == 0:
            modes[kernel.primary.mode] += 1
    return dict(modes), time.perf_counter() - started


def pct(counts: dict, key: str) -> float:
    total = sum(counts.values()) or 1
    return 100.0 * counts.get(key, 0) / total


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_baseline(atlas, seconds: float) -> None:
    """An undisturbed session, which has two distinct regimes.

    A fly starts hungry, so the first few minutes are spent foraging; only once energy
    is replenished does it settle into the rest-and-groom repertoire the README quotes
    for a ten-minute session. Both windows are checked, against their own expectation.
    """
    settled = seconds >= 300.0
    print(f"\nBaseline - {seconds:.0f} s, no interventions "
          f"({'settled' if settled else 'foraging'} regime)")
    kernel = build(atlas, environment="kitchen")
    modes, wall = run(kernel, seconds, sample=True)
    stats = kernel.telemetry.stats
    feeds, bouts = stats["feeds"], stats["groomingEvents"]
    mix = {key: pct(modes, key) for key in modes}

    print("    mode mix      " + ", ".join(
        f"{key} {value:.0f}%" for key, value in sorted(mix.items(), key=lambda kv: -kv[1])))
    print(f"    distance      {kernel.primary.distance_travelled:.0f} mm")
    print(f"    feeds         {feeds}   grooming bouts {bouts}")
    print(f"    wall          {wall:.1f} s")

    check(feeds >= 1, "the fly finds and takes at least one meal", f"{feeds} feeds")
    if settled:
        check(45 <= mix.get("REST", 0) <= 75, "settled session is rest-dominated",
              f"REST {mix.get('REST', 0):.0f}%")
        check(mix.get("GROOM", 0) >= 15, "grooming is a real part of the repertoire",
              f"GROOM {mix.get('GROOM', 0):.0f}%")
        check(bouts >= 8, "grooming bouts recur through the session", f"{bouts} bouts")
    else:
        # Hungry fly: foraging leads, but resting still punctuates it. If REST vanished
        # entirely the minimum-dwell logic would be broken.
        check(mix.get("WALK", 0) >= 30, "hungry fly forages", f"WALK {mix.get('WALK', 0):.0f}%")
        check(mix.get("REST", 0) >= 15, "foraging is punctuated by rest",
              f"REST {mix.get('REST', 0):.0f}%")
    kernel.shutdown()


def scenario_escape(atlas, seconds: float) -> None:
    """Each threat should produce repeated escapes without a self-retriggering runaway."""
    print(f"\nEscape reflex - {seconds:.0f} s per threat, stimulus at {THREAT_AT}")
    for key in ("swatter", "hand", "predator"):
        kernel = build(atlas, environment="kitchen")
        kernel._cmd_add_stimulus({"key": key, "x": THREAT_AT[0], "y": THREAT_AT[1],
                                  "z": THREAT_AT[2]})
        modes, _ = run(kernel, seconds, sample=True)
        escapes = kernel.primary.escape_events
        share = pct(modes, "ESCAPE")
        print(f"    {key:<10} escapes {escapes:<4} time in ESCAPE {share:.0f}%")
        check(escapes >= 15, f"{key}: repeated escapes", f"{escapes} escapes")
        # The refractory period exists precisely to stop this becoming a seizure.
        check(share <= 30, f"{key}: no escape runaway", f"ESCAPE {share:.0f}% of the time")
        kernel.shutdown()


def scenario_conditioning(atlas) -> None:
    """Wild type learns to avoid the paired odour; rutabaga cannot learn at all."""
    print("\nAssociative learning - peppermint paired with shock, 10 trials")
    trials, settle = 10, 0.5

    def learn(rutabaga: bool) -> float:
        kernel = build(atlas, environment="biolab")
        kernel._cmd_set_mutation({"key": "rutabaga", "value": rutabaga})
        for _ in range(trials):
            kernel._cmd_condition({"cs": "peppermint", "us": "shock"})
            run(kernel, settle)
        value = kernel.memory.value("peppermint")
        kernel.shutdown()
        return value

    wild = learn(False)
    mutant = learn(True)
    print(f"    wild type value   {wild:+.3f}")
    print(f"    rutabaga  value   {mutant:+.3f}")

    check(wild <= -0.80, "wild type learns a strong aversion", f"V = {wild:+.3f}")
    check(abs(mutant) <= 0.02, "rutabaga acquires nothing", f"V = {mutant:+.3f}")


def scenario_performance(atlas, seconds: float, runs: int = 3) -> None:
    """The kernel must stay far inside its 120 Hz budget."""
    print(f"\nPerformance - {runs} x {seconds:.0f} s runs")
    shares = []
    for index in range(runs):
        kernel = build(atlas, environment="kitchen")
        _, wall = run(kernel, seconds)
        share = 100.0 * wall / seconds
        shares.append(share)
        print(f"    run {index + 1}          {seconds:.0f} s simulated in {wall:.1f} s wall"
              f"  ({share:.1f}% of realtime)")
        kernel.shutdown()

    worst = max(shares)
    check(worst <= 30, "kernel stays well inside the realtime budget",
          f"worst {worst:.1f}% of the budget ({100.0 / worst:.1f}x realtime)")


def scenario_determinism(atlas, seconds: float) -> None:
    """A fixed step means the same input must produce the same trajectory."""
    print(f"\nDeterminism - two identical {seconds:.0f} s runs")
    traces = []
    for _ in range(2):
        kernel = build(atlas, environment="kitchen")
        run(kernel, seconds)
        traces.append(round(kernel.primary.distance_travelled, 6))
        kernel.shutdown()
    print(f"    distance       {traces[0]:.3f} mm then {traces[1]:.3f} mm")
    check(traces[0] == traces[1], "identical runs give identical physics",
          f"delta {abs(traces[0] - traces[1]):.9f} mm")


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the behaviour claimed in README.md")
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--quick", action="store_true",
                        help="short baseline, and skip the performance sweep")
    args = parser.parse_args()

    atlas = load_or_build(args.root, progress=lambda message: print(f"  ... {message}"))

    baseline_s = 180.0 if args.quick else 600.0
    threat_s = 90.0 if args.quick else 180.0

    scenario_baseline(atlas, baseline_s)
    scenario_escape(atlas, threat_s)
    scenario_conditioning(atlas)
    scenario_determinism(atlas, 60.0)
    if not args.quick:
        scenario_performance(atlas, 180.0)

    failed = [label for ok, label, _ in results if not ok]
    print(f"\n  {len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("  failed:")
        for label in failed:
            print(f"    - {label}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
