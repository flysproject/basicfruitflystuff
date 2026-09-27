# FlyBrain Sandbox

An autonomous *Drosophila melanogaster* driven by simulated neuromodulators, a
mushroom-body memory and a region-level connectome — rendered inside a real
environment you can drop stimuli into, with a live 3D brain HUD built from actual
Virtual Fly Brain imaging data.

```bash
python run.py
```

That is the whole setup. No dependency install, no virtualenv, no build step: the
backend is Python's standard library and the frontend is vendored Three.js, served
from the same process.

---

## Contents

- [What is real and what is simulated](#what-is-real-and-what-is-simulated)
- [Requirements](#requirements)
- [Running it](#running-it)
- [Architecture](#architecture)
- [How the fly decides](#how-the-fly-decides)
- [The atlas pipeline](#the-atlas-pipeline)
- [Extending it](#extending-it)
- [HTTP API](#http-api)
- [Verifying a checkout](#verifying-a-checkout)
- [Project layout](#project-layout)
- [Known limits](#known-limits)

---

## What is real and what is simulated

The app is deliberately explicit about the boundary, because the interesting part of
a system like this is knowing which claims it is actually making.

**Real, read from the source volumes at runtime:**

| Thing | Source |
| --- | --- |
| Brain template, 1210 × 566 × 174 | `VFB_00101567_(JRC2018Unisex).nrrd` — the JRC2018Unisex adult brain template |
| Registered binary mask, 452,342 voxels | `VFB_00102212_(VES_on_JRC2018Unisex_adult_brain).nrrd` — a real registered annotation, drawn as the violet accent in the brain HUD |
| Physical scale | Derived from the NRRD `space directions`: 627.9 × 293.7 × 174.0 µm, matching published JRC2018U |
| Axis orientation | Derived at bake time by a mirror-symmetry test, not assumed (see [The atlas pipeline](#the-atlas-pipeline)) |

**Approximated, and the app says so in the interface:**

- The eight neuropil ROIs (`SEZ`, `AL`, `OL`, `LH`, `MB`, `CX`, `GF`, `CC`) are
  **ellipsoids seeded from anatomical priors**, then snapped onto the template's own
  tissue mass. They are not a segmentation. Every one is drawn as a toggleable
  overlay, and each carries a measured tissue-overlap figure in its tooltip so you
  can see how well it lands.
- The connectome is a **19-edge qualitative region circuit**, not a synaptic
  reconstruction. Real FlyWire / Hemibrain data plugs in through a documented JSON
  format (see [Extending it](#extending-it)) without touching any simulation code.
- Neurotransmitter and hormone dynamics are first-order kinetic stand-ins chosen to
  have the right *qualitative* sign and time constant, not fitted parameters.

---

## Requirements

- **Python 3.10+** (developed on 3.12). Standard library only.
- A **WebGL2** browser. For a chromeless desktop window instead of a tab, one of:
  - `pip install pywebview` (uses the OS webview — smallest option), or
  - `pip install PySide6` (bundles Qt WebEngine, ~77 MB wheel).

Both are optional. Without either, the app opens in your default browser and behaves
identically.

Node.js is **not** required to run the app. It is only used if you want to re-vendor
Three.js.

---

## Running it

```bash
python run.py                              # open a window and start simulating
python run.py --no-window                  # serve only; open the printed URL yourself
python run.py --environment garden         # start in the Outdoor Garden
python run.py --flies 6                    # swarm of six, competing for the same food
python run.py --connectome my-circuit.json # load your own region circuit
python run.py --port 9000 --host 0.0.0.0   # reachable from another machine on the LAN
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--root` | this directory | Workspace holding the NRRD volumes |
| `--environment` | `kitchen` | Starting environment: `kitchen`, `garden`, `trashbin`, `biolab` |
| `--flies` | `1` | Number of flies, 1–10 |
| `--port` | `8777` | Preferred port; falls back to any free port and prints it |
| `--connectome` | built-in circuit | Path to a region-circuit JSON file |
| `--host` | `127.0.0.1` | Bind address |
| `--no-window` | off | Serve without opening a window |
| `--headless` | off | Run the kernel only, print a behavioural report, exit |
| `--headless-seconds` | `60` | Simulated seconds for `--headless` |

On the first run the atlas is baked from the NRRD volumes — about five seconds — and
cached under `cache/`. Every later start is instant. Edit `cache/rois.json` to move a
region and the cache is rebuilt automatically (the fingerprint covers file sizes and
modification times).

### Controls

| Input | Action |
| --- | --- |
| Click a toolbelt stimulus, then click the scene | Place that stimulus where you clicked |
| Drag / wheel | Orbit and zoom the camera |
| `1` `2` `3` | Follow, Tactical room, Fly vision cameras |
| `Space` | Pause / resume |
| `H` | Open the guide |
| `E` | Scenarios menu |
| `Esc` | Cancel an armed tool |

---

## Architecture

Three layers, communicating over loopback HTTP. Nothing is shared in-process across
the language boundary, which is what keeps the physics free of frame-rate coupling.

```
┌─────────────────── Python, one process ───────────────────┐
│  flybrain.kernel    120 Hz fixed-step loop, own thread     │
│      ├─ environment   analytic odour / thermal / light     │
│      ├─ flies         physics + behaviour FSM              │
│      ├─ endocrine     6 neuromodulator pools + physiology  │
│      ├─ connectome    region circuit, plasticity, memory   │
│      └─ telemetry     ring buffers, event log, exports     │
│  flybrain.server    http.server: static, SSE, POST, blobs  │
└───────────────────────────────────────────────────────────┘
             │ SSE state stream (60 Hz)      ▲ POST commands
             ▼                                │
┌───────────────── WebGL2 frontend ────────────────────────┐
│  scene.js   procedural environments, DoF + bloom composite │
│  fly.js     procedural anatomy, animation from state only  │
│  brain.js   raymarched brain HUD over real 3D textures     │
│  ui.js      manifest-driven panels, graphs, log, guide     │
└──────────────────────────────────────────────────────────┘
```

**Why the simulation is not in JavaScript.** A fixed 120 Hz step with real
accumulator semantics is what keeps behaviour reproducible and independent of the
display refresh rate. Running it in the page would tie the fly's decision rate to
whatever the compositor happened to give us that frame.

**Why SSE and not WebSocket.** The traffic is genuinely one-directional: 60 state
snapshots a second down, occasional user actions up. `EventSource` reconnects on its
own after a backend restart, needs no handshake or framing library, and is a few
lines of `http.server`. Commands go back over ordinary POSTs.

**Why the payload is split by rate.** The hot fields (fly kinematics, endocrine
levels, region activation) ride every frame. Heavy fields — environment geometry,
plume puffs, telemetry windows, the log tail — ride at a fraction of that (~10 Hz and
~5 Hz), because nothing in them changes fast enough to matter and re-sending them 60
times a second would dominate the frame budget.

**The renderer interpolates between the two most recent snapshots**, so a late or
coalesced frame cannot produce a visible stutter, and headings interpolate on the
shortest arc so a heading wrap does not spin the model.

### Measured performance

A headless 180-second run takes about 2.9 s wall: **~1.6% of the realtime budget,
roughly 60× faster than realtime.** Three consecutive runs give 1.6%, 1.6%, 1.6%.
An instrumented breakdown of the same workload attributes 1.50% to the step function
itself and 0.22% to the per-region sampling the headless report adds on top. It covers
the full cost: the advected odour field and every other analytic field, region routing,
endocrine integration, threat tracking and telemetry.

**Measure this on an otherwise idle machine.** With the app open in a browser alongside
— a WebGL scene rendering at 1728 × 1000 plus a 60 Hz SSE stream — the identical run can
report 8–14%, because the timing loop is competing for the same cores. That is
contention, not kernel cost. The live status bar reads anywhere from a few percent to
about 30% for the same reason: it measures the step from inside a process that is
concurrently serving the app, pushing state and answering the browser.

Two properties matter more than the headline number:

- **It is comfortably realtime.** A 120 Hz decision rate costs about 1.6% of one core,
  leaving the rest for the renderer and the transport.
- **It is deterministic.** Repeated runs produce an identical mode mix and an identical
  distance travelled (to nine decimal places — see `verify.py`), because the step is
  fixed and a wall-clock hiccup can never change the physics.

---

## How the fly decides

Every step, in order:

1. **Sense.** Sample the odour field, contact chemistry, temperature, light and any
   looming threat at the fly's actual position.
2. **Route.** Propagate activation through the region circuit (olfactory → mushroom
   body / lateral horn, optic lobe → giant fibre, and so on) with neuromodulator
   gains applied at each synapse.
3. **Decide.** A behaviour FSM with minimum dwell times per mode, plus preemptive
   overrides for escape, feeding and sleep.
4. **Drive.** Convert the decision into motor commands through endocrine-derived
   policy coefficients (`motor_gain`, `reactivity`, `persistence`, `risk_tolerance`,
   `flight_readiness`, `feeding_drive`).
5. **Animate.** Emit gait phase, wing phase/amplitude/frequency, proboscis extension
   and antenna swipe. **All animation comes from state** — the frontend never
   animates independently, so what you see is what the simulation computed.

### The odour field

Odour is modelled as **advected Gaussian puffs** rather than a diffusion solve. Points
of note, each of which was a measured failure before it was a design decision:

- A **near-field kernel** plus a **broad room-scale accumulation term** per odour. The
  broad term is why a fly can smell food across the room at all; the near field is
  what makes arrival a discrete event.
- The ceiling fan models a **floor jet** — radial outflow hugging the surface — not
  just downwash. Without it, odour was transported straight *through* the counter and
  the fly could not smell the food on top of it.
- **Chemotaxis is finite-difference gradient ascent on a motivational-value field**,
  not a two-antenna comparison. A fan-circulated plume is smooth over tens of
  millimetres, so a realistic 6 mm antenna baseline read as noise; the probe baseline
  is now 25 mm. A `plume_confidence` hysteresis (rise 3.0/s, decay 0.17/s, ~6 s of
  memory) lets the fly cast to re-acquire a lost plume instead of giving up.
- **Foraging is gated on hunger** (`hunger > 0.22 or NPF > 0.72`). Without the gate,
  room-scale odour made the fly forage permanently and it never groomed or rested.

### Escape

Looming is the derivative of angular size, computed with explicit velocities in mm/s.
A **3.0–4.5 s refractory period** is mandatory: an escape pulse raises octopamine,
which raises reactivity and lowers the threshold, and without the refractory the fly
retriggered itself and spent 48–83% of its time in `ESCAPE`.

Measured 180 s runs, one threat stimulus, one fly:

| Stimulus | Escapes | Time in `ESCAPE` |
| --- | --- | --- |
| Fly swatter | 47 | 9% |
| Human hand | 46 | 9% |
| Predator motion | 48 | 9% |

### Learning

Associative odour memory is a Rescorla–Wagner rule over the mushroom body. Odour value
is threaded explicitly through the decision path rather than held in a global.

Verified conditioning, peppermint paired with shock, 10 trials:

- Wild type → odour value **−0.933**
- `rutabaga` mutant → **exactly 0.0** (acquisition gain is zero, so nothing is learned)

That pair is the test that the mutation toggles are wired to the actual learning rule
rather than to a display string.

### A baseline session

600 simulated seconds, no interventions, one fly:

```
mode mix      REST 56%, GROOM 27%, WALK 16%, FEED 1%
distance      1931 mm
feeds         2      grooming bouts 18
physiology    energy 0.77   hunger 0.14   fear 0.00
region peaks  CC 0.77, MB 0.74, AL 0.69, CX 0.52, SEZ 0.41, LH 0.31, OL 0.27, GF 0.21
```

The fly rests and grooms for the large majority of an undisturbed session and feeds
twice, which is the repertoire you want to see *before* dropping anything into the
scene. Region activity concentrates in the higher-order integrative centres while the
sensory regions idle, because nothing is happening.

---

## The atlas pipeline

`flybrain/atlas.py` turns the NRRD volumes into everything the rest of the app uses:
a decimated template, a region-id volume, a mask, and the ROI table.

**Orientation is established, not assumed.** JRC2018Unisex is a template averaged from
124 images including left–right flips, so it is near-perfectly mirror symmetric. Baking
scores each axis with a mirror test on a coarse grid. `python -m flybrain.atlas
--inspect .` prints:

```
mirrorAsymmetry   { "x": 15.6, "y": 53.96, "z": 56.07 }
midlineAxis       "x"
extentUm          [ 627.89, 293.71, 174.0 ]
```

x is the clear outlier, so **x is left–right, y is anterior–posterior, z is
dorsal–ventral.** The absolute percentages shift with the scoring grid — on the
slightly finer grid the bake uses, x drops to 0.8% against 25.8% for y and 8.8% for z,
with the x centroid at 0.499 of the axis — but the ordering is unambiguous at every
resolution, which is the whole point of measuring rather than assuming. The physical
extent computed from the header (627.89 × 293.71 × 174.0 µm) independently matches
published JRC2018U, which is the cross-check that the header was read correctly.

**Two decimations, for two different jobs:**

| Purpose | Factor | Grid | Why |
| --- | --- | --- | --- |
| Template / mask textures | `(4, 4, 2)` | 303 × 142 × 87 | 3.7 MB per channel — instant over loopback, plenty for a raymarched HUD |
| Region classification | `(8, 8, 4)` | — | 475 k voxels; keeps the bake well under a second, and soft trilinear region boundaries read as a heatmap bloom, which is the desired look |

**The ROI table is checked against reality.** After snapping, each region's overlap with
the template's own tissue mass is measured. Before snapping, `SEZ` and `AL` sat at 0.0%
— they were floating in empty space and would have glowed in nothing. After:

| Region | On tissue | Region | On tissue |
| --- | --- | --- | --- |
| MB | 99.8% | LH | 89.4% |
| CX | 98.7% | GF | 89.2% |
| AL | 94.6% | OL | 85.4% |
| CC | 85.3% | SEZ | 80.6% |

Those figures are surfaced in the ROI legend tooltips, so the honesty is visible in the
interface rather than buried in a build log.

---

## Extending it

Every extension point is a file or a JSON payload — none require editing simulation
code.

### Move a neuropil — `cache/rois.json`

Written by `python -m flybrain.atlas . --write-rois` (or hand-authored). Coordinates
are normalised 0..1 of the volume, so the table is resolution independent:

```json
{
  "regions": [
    {
      "key": "MB",
      "label": "Mushroom Body",
      "blurb": "Associative learning centre...",
      "color": [192, 132, 252],
      "lobes": [[0.50, 0.62, 0.55, 0.09, 0.13, 0.10]],
      "layer": "integrative",
      "drives": ["CX"]
    }
  ]
}
```

Paired neuropils simply carry two lobes. The cache fingerprint covers this file, so
editing it rebuilds the bake on the next start.

### Plug in a real connectome — `--connectome circuit.json`

`JsonCircuitSource` accepts a pre-aggregated region circuit:

```json
{
  "name": "FlyWire region circuit",
  "note": "aggregated from 139k neurons on 2026-03-11",
  "rates": { "OL": 16.0, "GF": 22.0, "MB": 1.6 },
  "edges": [
    { "src": "AL", "dst": "MB", "weight": 0.83, "kind": "excitatory", "label": "PN -> KC" }
  ]
}
```

Region keys must match the atlas keys (`AL`, `OL`, `MB`, `CX`, `LH`, `SEZ`, `GF`,
`CC`). **Unknown keys are kept but simply never light up**, so a partial file is safe
to load. The status bar shows which source is live.

If you have whole-brain FlyWire or Hemibrain data rather than a pre-aggregated table,
`flybrain.connectome.aggregate_by_region(neurons, ...)` is the hook that folds
per-neuron records into region rates and edges — that is where a large reconstruction
gets reduced, and it is a pure function.

### Everything else

| Want to change | Where |
| --- | --- |
| A neuromodulator's name, colour, tooltip or decay | `MODULATORS` in `flybrain/neuro.py` |
| An environment's geometry, light, fog, wind, ambient odours | `ENVIRONMENTS` in `flybrain/env.py` |
| A stimulus's kind, colour, blurb or initial parameters | `STIMULUS_CATALOG` in `flybrain/env.py` |
| A mutation's gene, effect or gain | `MUTATION_INFO` and the gains in `flybrain/connectome.py` |
| A scripted scenario | `PRESET_SCENARIOS` in `flybrain/kernel.py` |
| Which channels the telemetry panel can plot | `SERIES_SCHEMA` in `flybrain/telemetry.py` |

The frontend builds its sliders, toolbelt, mutation toggles and legends **from the
backend manifest**, so adding a modulator or stimulus to the Python side makes it
appear in the interface with its real tooltip text. There is no second list to keep in
sync.

---

## HTTP API

All endpoints are on the same origin as the app.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | The application |
| `GET` | `/api/bootstrap` | Manifest: atlas, modulators, modes, environments, stimuli, presets, mutations, series |
| `GET` | `/api/snapshot` | Full current state, one-shot |
| `GET` | `/api/stream` | SSE state stream, 60 Hz |
| `GET` | `/api/atlas/template.bin` | Decimated template volume, uint8 |
| `GET` | `/api/atlas/region-ids.bin` | Region-id volume, uint8 |
| `GET` | `/api/atlas/mask.bin` | Registered mask, uint8 (404 if absent) |
| `GET` | `/api/log?since=N` | Log entries after cursor `N` |
| `GET` | `/api/export/csv` | Telemetry download |
| `GET` | `/api/export/json` | Full session download |
| `GET` | `/api/profile` | Current endocrine target profile |
| `POST` | `/api/command` | `{"name": "...", "payload": {...}}` |

Commands are queued and applied on the kernel thread, so nothing mutates simulation
state from an HTTP thread. The full set: `set_slider`, `apply_profile`, `set_mutation`,
`load_environment`, `add_stimulus`, `remove_stimulus`, `move_stimulus`, `set_param`,
`clear_stimuli`, `set_paused`, `set_time_scale`, `set_day_length`, `set_fly_count`,
`condition`, `set_conditioning`, `forget`, `deliver`, `preset`, `reset`, `export`,
`export_profile`, `load_profile`.

---

## Verifying a checkout

Four checks, none of which need a browser:

```bash
python flybrain/nrrd.py                    # NRRD decoder round-trip, synthetic data
python -m flybrain.atlas --inspect .      # geometry + the orientation proof
python -m flybrain.atlas . --force        # rebuild the bake from scratch
python run.py --headless --headless-seconds 120
python verify.py                           # every behavioural claim on this page
```

The first check needs no input data: it writes a small synthetic `.nrrd` to a temp file
and round-trips it through the reader, asserting shape, spacing, voxel addressing,
strided decimation, decimation offsets and bounding boxes — the parts of the decoder
that are easy to get subtly wrong and hard to notice.

The headless run is the behavioural smoke test: it prints the simulated-vs-wall
speedup, the mode mix, distance travelled, escape and feeding counts, physiology,
per-region peaks and the tail of the event log. It drives the same step function the
app does, minus the transport, which makes it also useful for batch experiments.

**`verify.py` is the real gate.** Every number on this page — the settled mode mix, the
escape counts, the conditioned odour value, the wild-type-versus-`rutabaga` contrast,
the performance budget, and run-to-run determinism — is a check with an explicit
threshold, and the script exits non-zero if any of them drifts:

```
Baseline - 600 s, no interventions (settled regime)
    mode mix      REST 56%, GROOM 27%, WALK 16%, FEED 1%
  PASS  settled session is rest-dominated            REST 56%
  ...
  14/14 checks passed
```

`--quick` cuts the baseline to the foraging window, skips the performance sweep, and
finishes in about a minute, which is the right thing to run while iterating.

For the frontend, open the app and confirm the browser console is clean. Two classes
of defect are only visible by looking: a shader that fails to link (the brain HUD
draws nothing) and an element that cannot hide because an author `display` rule
outranks the `hidden` attribute.

---

## Project layout

```
run.py                     one-command launcher, window-shell detection, headless report
verify.py                  reproduces every behavioural claim in this README
flybrain/
  nrrd.py                  dependency-free NRRD reader (gzip, strided decimation)
  atlas.py                 ROI table, tissue snapping, overlap metric, bake cache
  neuro.py                 6 modulator pools, physiology, derived policy coefficients
  connectome.py            region circuit, pluggable sources, RW memory, mutations
  env.py                   analytic odour / thermal / light / threat fields, terrain
  fly.py                   physics, behaviour FSM, gradient chemotaxis, escape reflex
  telemetry.py             ring buffers, event log, CSV/JSON export
  kernel.py                120 Hz thread, command surface, presets, conditioning
  server.py                stdlib HTTP + SSE server, atlas blobs
web/
  index.html               panels, HUDs, toolbelt, guide
  css/app.css              dark scientific interface
  js/net.js                bootstrap, SSE client, snapshot interpolation
  js/scene.js              renderer, post-processing chain, cameras, lighting
  js/fly.js                procedural anatomy and animation
  js/world.js              environments and stimulus geometry
  js/brain.js              raymarched brain HUD over real 3D textures
  js/ui.js                 manifest-driven interface, graphs, log, audio
  js/main.js               wiring and the frame loop
  vendor/                  Three.js r186 + licence
cache/                     atlas.json, atlas.bin (regenerated), rois.json (your overrides)
exports/, profiles/        written by the export and profile commands
```

---

## Known limits

- **The ROIs are ellipsoids.** Eight per-lobe ellipsoids cannot express the true shape
  of the mushroom body calyx or the ellipsoid body. They are placed and validated
  against real tissue, but they are a stand-in for a segmentation. Supply
  `cache/rois.json` from a real annotation to fix this.
- **The connectome is qualitative.** 19 edges with hand-set weights reproduce the right
  regional causality and the right learning phenotype; they are not synapse counts.
- **Modulator kinetics are not fitted.** Signs and time constants are chosen to be
  biologically plausible, not estimated from data.
- **Single-threaded physics.** All flies step on the one 120 Hz thread. This is far
  from saturated at ten flies, but it is a ceiling.
- **Windows, macOS and Linux all work**, but the chromeless-window path has only been
  exercised on Windows.
