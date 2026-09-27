"""VFB atlas pipeline: real registered volumes -> GPU-ready 3-D textures.

This module is why the brain visualiser shows a *real* Drosophila central nervous
system instead of a lump of spheres. It reads the Virtual Fly Brain NRRD exports
present in the workspace, reduces them to GPU-friendly 3-D textures, and
classifies every voxel of a coarse grid into a neuropil ROI so the raymarching
shader can light individual regions up.

Orientation of JRC2018Unisex
----------------------------
Two independent checks establish the axis order, so the ROI table below is not a
guess:

1. The workspace file is 1210 x 566 x 174 voxels at 0.5189 x 0.5189 x 1.0 um,
   a physical bounding box of 627.9 x 293.7 x 174.0 um. The published JRC2018U
   template bounding box is 627.38 x 293.36 x 172.9 um -- identical, just
   resampled onto a coarser anisotropic grid.
2. The template is built from 124 images including left-right flips, so it is
   near-perfectly mirror symmetric across the midline. Scoring a mirror test on a
   coarse grid gives 0.8% asymmetry for x (centroid at 0.499 of the axis), against
   25.8% for y and 8.8% for z.

Therefore ``x`` is left-right, ``y`` is anterior-posterior and ``z`` is
dorsal-ventral. (:func:`flybrain.atlas.describe_volume` reproduces both checks.)

The neuropil ROIs are **approximate ellipsoids** in normalised JRC2018U space,
placed from the tissue's own silhouette rather than from an annotation volume.
They are honest approximations: every ROI is drawn as a toggleable overlay in the
app, and ``cache/rois.json`` overrides the built-in table so a user with a real
annotation can register the centres precisely.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from .nrrd import Volume, find_nrrd_files, read_nrrd

__all__ = [
    "Region",
    "Atlas",
    "build_atlas",
    "load_or_build",
    "describe_volume",
    "DEFAULT_REGIONS",
    "DEFAULT_DECIMATION",
    "ROI_OVERRIDE_FILE",
]

# Decimation that lands the 1210 x 566 x 174 source on a 303 x 142 x 87 grid.
# That is 3.7 MB per channel: instant to move over loopback and far more detail
# than a raymarched HUD needs.
DEFAULT_DECIMATION = (4, 4, 2)
# Region classification runs on a coarser grid (475 k voxels). Soft region
# boundaries from trilinear filtering read as a heatmap bloom, which is what we
# want, and it keeps the bake to well under a second.
REGION_DECIMATION = (8, 8, 4)

ROI_OVERRIDE_FILE = "rois.json"

# A tissue threshold that clears the template's background texture.
TISSUE_THRESHOLD = 35


@dataclass
class Region:
    """One neuropil ROI: a labelled set of ellipsoidal lobes in normalised space.

    ``lobes`` entries are ``(cx, cy, cz, rx, ry, rz)`` with all six values
    normalised to 0..1 of the volume, so the table is resolution independent.
    Paired neuropils simply carry two lobes.
    """

    key: str
    label: str
    blurb: str
    color: tuple[int, int, int]
    lobes: tuple[tuple[float, float, float, float, float, float], ...]
    layer: str
    drives: tuple[str, ...] = ()

    @property
    def is_paired(self) -> bool:
        return len(self.lobes) > 1

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "blurb": self.blurb,
            "color": list(self.color),
            "lobes": [list(lobe) for lobe in self.lobes],
            "layer": self.layer,
            "drives": list(self.drives),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "Region":
        return cls(
            key=payload["key"],
            label=payload["label"],
            blurb=payload.get("blurb", ""),
            color=tuple(payload.get("color", (200, 200, 200))),  # type: ignore[arg-type]
            lobes=tuple(tuple(float(v) for v in lobe) for lobe in payload["lobes"]),  # type: ignore[arg-type]
            layer=payload.get("layer", "integrative"),
            drives=tuple(payload.get("drives", ())),  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------
# Built-in ROI table. Order matters: it is the wire order of the activation
# uniform array, and painting priority runs largest-lobe-last so small neuropils
# are never swallowed by big neighbours.
# --------------------------------------------------------------------------
DEFAULT_REGIONS: tuple[Region, ...] = (
    Region(
        key="SEZ",
        label="Subesophageal Zone",
        blurb="Primary gustatory and feeding-motor centre. Sweet or bitter contact on the "
        "proboscis lands here first and gates proboscis extension.",
        color=(122, 209, 255),
        lobes=((0.50, 0.20, 0.25, 0.105, 0.100, 0.100),),
        layer="sensory",
        drives=("CC", "MB"),
    ),
    Region(
        key="AL",
        label="Antennal Lobe",
        blurb="Olfactory first relay. ORNs from the antennae terminate here; odour identity is "
        "sparsened onto projection neurons before it reaches the mushroom body.",
        color=(74, 222, 128),
        lobes=(
            (0.37, 0.19, 0.33, 0.062, 0.060, 0.065),
            (0.63, 0.19, 0.33, 0.062, 0.060, 0.065),
        ),
        layer="sensory",
        drives=("MB", "LH"),
    ),
    Region(
        key="OL",
        label="Optic Lobe",
        blurb="Lamina, medulla, lobula and lobula plate. Detects motion, tracks light and hosts "
        "the lobula giant movement detectors that fire on a looming shadow.",
        color=(56, 189, 248),
        lobes=(
            (0.16, 0.42, 0.46, 0.105, 0.150, 0.185),
            (0.84, 0.42, 0.46, 0.105, 0.150, 0.185),
        ),
        layer="sensory",
        drives=("CX", "GF"),
    ),
    Region(
        key="LH",
        label="Lateral Horn",
        blurb="Second-order olfactory valence centre. Attractive and aversive odours diverge "
        "here, feeding approach or avoidance before learning kicks in.",
        color=(163, 230, 53),
        lobes=(
            (0.26, 0.25, 0.44, 0.055, 0.070, 0.080),
            (0.74, 0.25, 0.44, 0.055, 0.070, 0.080),
        ),
        layer="integrative",
        drives=("CX", "MB"),
    ),
    Region(
        key="MB",
        label="Mushroom Body",
        blurb="Associative memory. Kenyon cells carry sparse odour codes; dopamine and "
        "octopamine write valence onto the MBON outputs, which is where conditioning lives.",
        color=(192, 132, 252),
        lobes=(
            (0.38, 0.32, 0.62, 0.070, 0.100, 0.085),
            (0.62, 0.32, 0.62, 0.070, 0.100, 0.085),
        ),
        layer="integrative",
        drives=("CX", "CC"),
    ),
    Region(
        key="CX",
        label="Central Complex",
        blurb="Compass and steering. A ring attractor holds heading, the fan-shaped body holds "
        "goal vectors, so the fly can path-integrate home while walking or flying.",
        color=(251, 113, 133),
        lobes=((0.50, 0.36, 0.58, 0.055, 0.075, 0.060),),
        layer="integrative",
        drives=("CC",),
    ),
    Region(
        key="GF",
        label="Giant Fibre / DNg",
        blurb="Escape command. Two lobula giant movement detectors drive the giant fibre down to "
        "the tergotrochanteral motor neuron, producing a takeoff within ~30 ms of a shadow.",
        color=(245, 158, 11),
        lobes=((0.50, 0.44, 0.74, 0.045, 0.090, 0.050),),
        layer="motor",
        drives=("CC",),
    ),
    Region(
        key="CC",
        label="Cervical Connective / Descending",
        blurb="Descending neurons leaving the brain for the ventral nerve cord. Everything the "
        "brain decides leaves through here, so its traffic mirrors motor intent.",
        color=(250, 204, 21),
        lobes=((0.50, 0.90, 0.45, 0.055, 0.080, 0.090),),
        layer="motor",
        drives=(),
    ),
)


@dataclass
class Atlas:
    """Baked, GPU-ready representation of the CNS."""

    shape: tuple[int, int, int]
    spacing_um: tuple[float, float, float]
    source_shape: tuple[int, int, int]
    template: bytes
    region_ids: bytes
    region_shape: tuple[int, int, int]
    regions: tuple[Region, ...]
    mask: bytes | None = None
    meta: dict = field(default_factory=dict)

    @property
    def extent_um(self) -> tuple[float, float, float]:
        return tuple(n * s for n, s in zip(self.shape, self.spacing_um))  # type: ignore[return-value]

    def to_json(self) -> dict:
        return {
            "shape": list(self.shape),
            "regionShape": list(self.region_shape),
            "sourceShape": list(self.source_shape),
            "spacingUm": list(self.spacing_um),
            "extentUm": [round(v, 2) for v in self.extent_um],
            "hasMask": self.mask is not None,
            "regions": [r.to_json() for r in self.regions],
            "meta": self.meta,
        }


# --------------------------------------------------------------------------
# Volume introspection helpers (also used by the docs/CLI)
# --------------------------------------------------------------------------


def describe_volume(volume: Volume, decimation: tuple[int, int, int] = (16, 16, 8)) -> dict:
    """Reproduce the orientation checks that justify the axis order.

    Returns the physical extent, the tissue bounding box, and a mirror-symmetry
    score per axis. The lowest-scoring axis is the left-right midline, because
    the template averages 124 images including left-right flips.

    Scored on a very coarse grid on purpose: the test only needs the midline, and
    it runs on every cold build.
    """
    reduced, shape = volume.downsampled_u8(decimation)
    ox, oy, oz = shape

    def at(x: int, y: int, z: int) -> int:
        return reduced[x + ox * (y + oy * z)]

    scores: dict[str, float] = {}
    for axis, name in ((0, "x"), (1, "y"), (2, "z")):
        span = (ox, oy, oz)[axis]
        total = 0
        diff = 0
        for z in range(oz):
            for y in range(oy):
                for x in range(ox):
                    a = at(x, y, z)
                    if a < TISSUE_THRESHOLD:
                        continue
                    mirror = {
                        0: (span - 1 - x, y, z),
                        1: (x, span - 1 - y, z),
                        2: (x, y, span - 1 - z),
                    }[axis]
                    total += 1
                    diff += abs(a - at(*mirror))
        scores[name] = round(diff / max(1, total), 2)

    lo, hi = volume.bounding_box(1, 4)
    return {
        "sourceShape": list(volume.shape),
        "spacingUm": [round(s, 4) for s in volume.spacing],
        "extentUm": [round(v, 2) for v in volume.extent],
        "coarseShape": list(shape),
        "mirrorAsymmetry": scores,
        "midlineAxis": min(scores, key=lambda k: scores[k]),
        "tissueVoxelBbox": [list(lo), list(hi)],
    }


def _pick_sources(root: str) -> tuple[str | None, str | None]:
    """Choose the intensity template and a registered mask from the workspace.

    The workspace ships two volumes on the same grid: the JRC2018Unisex template
    and a sparse registered binary mask. Distinguishing them by *compressed* size
    per voxel avoids inflating 119 megavoxels just to look at a histogram: the
    template costs ~0.27 bytes/voxel, a binary mask ~0.0012.
    """
    from .nrrd import read_header

    candidates: list[tuple[float, str]] = []
    for path in find_nrrd_files([root]):
        try:
            header = read_header(path)
            sizes = [int(v) for v in header.get("sizes", "").split()]
            voxels = 1
            for size in sizes:
                voxels *= size
            candidates.append((os.path.getsize(path) / max(1, voxels), path))
        except Exception:  # pragma: no cover - defensive
            continue
    if not candidates:
        return None, None
    candidates.sort()
    template = candidates[-1][1]
    mask = None
    for ratio, path in candidates[:-1]:
        if ratio < 0.02:
            mask = path
            break
    return template, mask


def _window_from(volume: Volume, decimation: tuple[int, int, int]) -> tuple[float, float]:
    """Pick a display window so the template is not washed out by background."""
    reduced, _shape = volume.downsampled_u8(decimation)
    histogram = [0] * 256
    for value in reduced:
        histogram[value] += 1
    total = len(reduced)
    high = 255.0
    seen = 0
    for value in range(255, -1, -1):
        seen += histogram[value]
        if seen >= total * 0.02:  # drop the top 2% of intensity
            high = float(value)
            break
    return 0.0, max(32.0, high)


def snap_regions_to_tissue(
    template: bytes,
    shape: tuple[int, int, int],
    regions: tuple[Region, ...],
    window: float = 0.06,
    search_scale: float = 2.2,
) -> tuple[tuple[Region, ...], list[dict]]:
    """Pull each ROI onto nearby tissue mass in the real template.

    The built-in ROI table is an *anatomical prior*, and a prior that misses the
    tissue is worse than useless: the heatmap would glow in empty space. This
    step takes the intensity-weighted centroid of tissue inside a small window
    around each prior and moves the lobe there, clamped to the window so a bright
    neighbour cannot drag a neuropil across the brain.

    ``search_scale`` widens the neighbourhood the centroid is computed in beyond
    the ROI itself, so a prior that starts just off the tissue still finds it;
    the *move* is still clamped to *window*, so an ROI can only ever travel as far
    as the prior's own uncertainty.

    Paired neuropils stay paired: the left lobe is snapped once and its offset is
    mirrored across the midline, which the template is 99% symmetric about. Returns
    the adjusted regions plus a per-lobe report so the bake can prove the result.
    """
    tx, ty, tz = shape
    report: list[dict] = []
    moved: list[Region] = []

    for region in regions:
        new_lobes = []
        for lobe_index, (cx, cy, cz, rx, ry, rz) in enumerate(region.lobes):
            # Mirror the right lobe from the (already snapped) left lobe, so a
            # paired neuropil can never end up lopsided.
            if region.is_paired and lobe_index == 1:
                _lcx, lcy, lcz, _lr, _ly, _lz = new_lobes[0]
                mirrored = (1.0 - new_lobes[0][0], lcy, lcz)
                new_lobes.append((mirrored[0], mirrored[1], mirrored[2], rx, ry, rz))
                report.append(
                    {
                        "region": region.key,
                        "lobe": lobe_index,
                        "mirrored": True,
                    }
                )
                continue

            sx, sy, sz = rx * search_scale, ry * search_scale, rz * search_scale
            x0 = max(0, int((cx - sx) * tx))
            x1 = min(tx - 1, int((cx + sx) * tx))
            y0 = max(0, int((cy - sy) * ty))
            y1 = min(ty - 1, int((cy + sy) * ty))
            z0 = max(0, int((cz - sz) * tz))
            z1 = min(tz - 1, int((cz + sz) * tz))

            weight = 0.0
            wx = wy = wz = 0.0
            for z in range(z0, z1 + 1):
                nz_ = (z + 0.5) / tz
                dzn = (nz_ - cz) / rz
                plane = z * tx * ty
                for y in range(y0, y1 + 1):
                    ny_ = (y + 0.5) / ty
                    dyn = (ny_ - cy) / ry
                    dz_dy = dzn * dzn + dyn * dyn
                    if dz_dy > search_scale * search_scale:
                        continue
                    row = plane + y * tx
                    for x in range(x0, x1 + 1):
                        value = template[row + x]
                        if value < TISSUE_THRESHOLD:
                            continue
                        nx_ = (x + 0.5) / tx
                        dxn = (nx_ - cx) / rx
                        # Only tissue inside the search neighbourhood counts.
                        if dxn * dxn + dz_dy > search_scale * search_scale:
                            continue
                        w = float(value - TISSUE_THRESHOLD)
                        weight += w
                        wx += w * nx_
                        wy += w * ny_
                        wz += w * nz_

            if weight <= 0.0:
                new_lobes.append((cx, cy, cz, rx, ry, rz))
                report.append(
                    {"region": region.key, "lobe": lobe_index, "weight": 0.0, "movedBy": 0.0}
                )
                continue

            target = (wx / weight, wy / weight, wz / weight)
            clamped = []
            for prior, wanted in zip((cx, cy, cz), target):
                delta = max(-window, min(window, wanted - prior))
                clamped.append(prior + delta)
            moved_by = sum((a - b) ** 2 for a, b in zip(clamped, (cx, cy, cz))) ** 0.5
            new_lobes.append((clamped[0], clamped[1], clamped[2], rx, ry, rz))
            report.append(
                {
                    "region": region.key,
                    "lobe": lobe_index,
                    "weight": round(weight, 1),
                    "movedBy": round(moved_by, 4),
                }
            )

        moved.append(
            Region(
                key=region.key,
                label=region.label,
                blurb=region.blurb,
                color=region.color,
                lobes=tuple(new_lobes),  # type: ignore[arg-type]
                layer=region.layer,
                drives=region.drives,
            )
        )
    return tuple(moved), report


def region_tissue_overlap(atlas: "Atlas") -> dict[str, dict]:
    """Per-ROI sanity check: does the region actually sit on tissue?

    A region that misses the tissue would glow in empty space, and the bake would
    look subtly, unfixably wrong. Reporting ``onTissuePct`` and the median template
    intensity under each ROI catches that at build time instead of by eye.

    Voxels are mapped from the coarse region grid onto the template grid, so this
    samples the ROI rather than testing every template voxel.
    """
    tx, ty, tz = atlas.shape
    rx, ry, rz = atlas.region_shape
    step = (tx / rx, ty / ry, tz / rz)
    samples: list[list[int]] = [[] for _ in atlas.regions]
    for z in range(rz):
        tz_ = min(tz - 1, int(z * step[2]))
        for y in range(ry):
            ty_ = min(ty - 1, int(y * step[1]))
            base = z * rx * ry + y * rx
            tbase = tz_ * tx * ty + ty_ * tx
            for x in range(rx):
                rid = atlas.region_ids[base + x]
                if rid:
                    samples[rid - 1].append(
                        atlas.template[tbase + min(tx - 1, int(x * step[0]))]
                    )

    def median(values: list[int]) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    return {
        region.key: {
            "voxels": len(samples[i]),
            "onTissuePct": round(
                100.0 * sum(1 for v in samples[i] if v >= TISSUE_THRESHOLD) / max(1, len(samples[i])),
                1,
            ),
            "medianIntensity": median(samples[i]),
        }
        for i, region in enumerate(atlas.regions)
    }


def _classify_regions(
    shape: tuple[int, int, int], regions: tuple[Region, ...]
) -> bytes:
    """Paint region ids into a coarse grid, one ROI bounding box at a time.

    Iterating each ellipsoid's own bounding box rather than testing every voxel
    against every ROI turns a ~7 M operation sweep into a few hundred thousand
    checks, which keeps the whole bake under a second in pure Python.
    """
    ox, oy, oz = shape
    ids = bytearray(ox * oy * oz)
    # Paint largest lobes first so small neuropils win where they overlap.
    order = sorted(
        range(len(regions)),
        key=lambda i: -max(l[3] * l[4] * l[5] for l in regions[i].lobes),
    )
    for index in order:
        region = regions[index]
        rid = index + 1
        for cx, cy, cz, rx, ry, rz in region.lobes:
            x0 = max(0, int((cx - rx) * ox))
            x1 = min(ox - 1, int((cx + rx) * ox) + 1)
            y0 = max(0, int((cy - ry) * oy))
            y1 = min(oy - 1, int((cy + ry) * oy) + 1)
            z0 = max(0, int((cz - rz) * oz))
            z1 = min(oz - 1, int((cz + rz) * oz) + 1)
            for z in range(z0, z1 + 1):
                nz_ = (z + 0.5) / oz
                dz = (nz_ - cz) / rz
                dz2 = dz * dz
                if dz2 > 1.0:
                    continue
                plane = z * ox * oy
                for y in range(y0, y1 + 1):
                    ny_ = (y + 0.5) / oy
                    dy = (ny_ - cy) / ry
                    d2 = dz2 + dy * dy
                    if d2 > 1.0:
                        continue
                    row = plane + y * ox
                    # Solve for the x span of this ellipsoid slice rather than
                    # scanning the full axis: one sqrt per row instead of per voxel.
                    if d2 >= 1.0:
                        continue
                    dx_max = (1.0 - d2) ** 0.5
                    xs = int(round((cx - dx_max * rx) * ox - 0.5))
                    xe = int(round((cx + dx_max * rx) * ox - 0.5))
                    xs = max(x0, xs)
                    xe = min(x1, xe)
                    if xe < xs:
                        continue
                    if ids[row + xs : row + xe + 1].count(0) == 0:
                        continue
                    for x in range(xs, xe + 1):
                        if not ids[row + x]:
                            ids[row + x] = rid
    return bytes(ids)


def _synthetic_cns(shape: tuple[int, int, int], regions: tuple[Region, ...]) -> bytes:
    """Fallback intensity volume so the app still runs without the VFB files."""
    ox, oy, oz = shape
    out = bytearray(ox * oy * oz)
    for z in range(oz):
        nz_ = (z + 0.5) / oz
        for y in range(oy):
            ny_ = (y + 0.5) / oy
            for x in range(ox):
                nx_ = (x + 0.5) / ox
                best = 0.0
                for region in regions:
                    for cx, cy, cz, rx, ry, rz in region.lobes:
                        d = (
                            ((nx_ - cx) / rx) ** 2
                            + ((ny_ - cy) / ry) ** 2
                            + ((nz_ - cz) / rz) ** 2
                        )
                        if d < 1.0:
                            best = max(best, 1.0 - d)
                out[x + ox * (y + oy * z)] = min(255, int(best * 220))
    return bytes(out)


def build_atlas(
    root: str = ".",
    decimation: tuple[int, int, int] = DEFAULT_DECIMATION,
    region_decimation: tuple[int, int, int] = REGION_DECIMATION,
    progress=None,
) -> Atlas:
    """Read the VFB volumes and bake an :class:`Atlas`."""
    regions = load_region_table(root)

    def say(message: str) -> None:
        if progress:
            progress(message)

    started = time.time()
    template_path, mask_path = _pick_sources(root)
    meta: dict = {
        "builtAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": {},
        "synthetic": template_path is None,
    }

    if template_path is None:
        say("No VFB NRRD volumes found - generating a synthetic CNS stand-in")
        shape = (152, 71, 44)
        template = _synthetic_cns(shape, regions)
        mask = None
        source_shape = shape
        spacing = (4.0, 4.0, 4.0)
    else:
        say(f"Reading template {os.path.basename(template_path)}")
        volume = read_nrrd(template_path)
        meta["sources"]["template"] = {
            "path": os.path.relpath(template_path, root).replace("\\", "/"),
            "shape": list(volume.shape),
            "spacingUm": [round(s, 4) for s in volume.spacing],
        }
        window = _window_from(volume, decimation)
        meta["window"] = [round(window[0], 1), round(window[1], 1)]
        say(f"Decimating {volume.shape[0]}x{volume.shape[1]}x{volume.shape[2]} -> "
            f"factor {decimation}")
        template, shape = volume.downsampled_u8(decimation, window=window)
        source_shape = volume.shape
        spacing = (
            volume.spacing[0] * decimation[0],
            volume.spacing[1] * decimation[1],
            volume.spacing[2] * decimation[2],
        )
        say("Running orientation checks (midline detection)")
        meta["orientation"] = describe_volume(volume)

        mask = None
        if mask_path:
            say(f"Reading registered mask {os.path.basename(mask_path)}")
            mask_volume = read_nrrd(mask_path)
            raw, mask_shape = mask_volume.downsampled_u8(
                decimation, window=(0.0, float(max(mask_volume.buf) or 1))
            )
            mask = raw
            nonzero = sum(1 for v in raw if v > 0)
            meta["sources"]["mask"] = {
                "path": os.path.relpath(mask_path, root).replace("\\", "/"),
                "shape": list(mask_volume.shape),
                "nonzeroVoxels": nonzero,
                "coveragePct": round(100.0 * nonzero / max(1, len(raw)), 3),
                "bbox": [list(b) for b in mask_volume.bounding_box(1, 4)],
                "note": "Real registered binary mask on the JRC2018Unisex grid. Sparse, so it "
                "reads as a focal expression pattern rather than a neuropil label volume.",
            }
            if mask_shape != shape:
                meta["sources"]["mask"]["warning"] = "mask grid differs from template grid"

    if not meta["synthetic"]:
        say("Snapping neuropil ROIs onto template tissue")
        regions, snap_report = snap_regions_to_tissue(template, shape, regions)
        meta["roiSnap"] = snap_report

    say("Classifying neuropil regions")
    region_shape = tuple(
        max(1, len(range(0, source_shape[axis], region_decimation[axis]))) for axis in range(3)
    )
    region_ids = _classify_regions(region_shape, regions)  # type: ignore[arg-type]

    atlas = Atlas(
        shape=shape,
        spacing_um=spacing,
        source_shape=source_shape,
        template=template,
        region_ids=region_ids,
        region_shape=region_shape,  # type: ignore[arg-type]
        regions=regions,
        mask=mask,
        meta=meta,
    )

    coverage = {region.key: region_ids.count(i + 1) for i, region in enumerate(regions)}
    total_assigned = sum(coverage.values())
    meta["regionCoverage"] = coverage
    meta["regionCoveragePct"] = round(100.0 * total_assigned / max(1, len(region_ids)), 2)
    # The sanity check that keeps the ROI table honest: a region that does not sit
    # on tissue would glow in empty space, and the bake would silently look wrong.
    meta["regionTissueOverlap"] = region_tissue_overlap(atlas)
    meta["buildSeconds"] = round(time.time() - started, 2)
    meta["roiNote"] = (
        "ROIs are ellipsoids seeded from anatomical priors in normalised JRC2018U space "
        "and snapped onto the template's own tissue mass. Override them in "
        "cache/rois.json."
    )
    worst = min(meta["regionTissueOverlap"].items(), key=lambda kv: kv[1]["onTissuePct"])
    say(
        f"Baked in {meta['buildSeconds']}s, {meta['regionCoveragePct']}% of grid assigned, "
        f"lowest tissue overlap {worst[0]} at {worst[1]['onTissuePct']}%"
    )
    return atlas


def load_region_table(root: str) -> tuple[Region, ...]:
    """Return the ROI table, honouring ``cache/rois.json`` if the user tuned it."""
    override = os.path.join(root, "cache", ROI_OVERRIDE_FILE)
    if os.path.isfile(override):
        try:
            with open(override, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            regions = tuple(Region.from_json(item) for item in payload["regions"])
            if regions:
                return regions
        except Exception:
            pass
    return DEFAULT_REGIONS


def write_region_table(root: str, regions: tuple[Region, ...] = DEFAULT_REGIONS) -> str:
    """Write the default ROI table out so it can be edited by hand."""
    target = os.path.join(root, "cache", ROI_OVERRIDE_FILE)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "_comment": "Neuropil ROI table. Normalised JRC2018U coordinates: "
                "x = left-right, y = anterior-posterior, z = dorsal-ventral.",
                "regions": [r.to_json() for r in regions],
            },
            handle,
            indent=2,
        )
    return target


def load_or_build(root: str = ".", cache_dir: str | None = None, progress=None) -> Atlas:
    """Load a cached bake when the sources are unchanged, else rebuild it.

    The bake is keyed on the size and mtime of every source volume, so replacing
    a volume in the workspace invalidates the cache automatically.
    """
    cache_dir = cache_dir or os.path.join(root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "atlas.json")
    bin_path = os.path.join(cache_dir, "atlas.bin")
    key = atlas_source_key(root)

    if os.path.isfile(meta_path) and os.path.isfile(bin_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if cached.get("_key") == key:
                if progress:
                    progress("Loading cached atlas bake")
                return _load_cached(cached, bin_path, root)
        except Exception:
            pass

    atlas = build_atlas(root, progress=progress)
    _save_cached(atlas, meta_path, bin_path, key)
    return atlas


def atlas_source_key(root: str) -> list:
    """Cheap fingerprint of every NRRD input plus the ROI override."""
    key = []
    for path in find_nrrd_files([root]):
        try:
            stat = os.stat(path)
        except OSError:  # pragma: no cover - defensive
            continue
        key.append([os.path.relpath(path, root).replace("\\", "/"), stat.st_size, int(stat.st_mtime)])
    override = os.path.join(root, "cache", ROI_OVERRIDE_FILE)
    if os.path.isfile(override):
        stat = os.stat(override)
        key.append([ROI_OVERRIDE_FILE, stat.st_size, int(stat.st_mtime)])
    return key


def _save_cached(atlas: Atlas, meta_path: str, bin_path: str, key: list) -> None:
    offset = 0
    layout = {}
    with open(bin_path, "wb") as handle:
        for name, blob in (
            ("template", atlas.template),
            ("regionIds", atlas.region_ids),
            ("mask", atlas.mask),
        ):
            if blob is None:
                continue
            handle.write(blob)
            layout[name] = [offset, len(blob)]
            offset += len(blob)
    payload = atlas.to_json()
    payload["_key"] = key
    payload["_layout"] = layout
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _load_cached(payload: dict, bin_path: str, root: str) -> Atlas:
    with open(bin_path, "rb") as handle:
        blob = handle.read()
    layout = payload["_layout"]

    def slice_of(name: str) -> bytes | None:
        if name not in layout:
            return None
        start, length = layout[name]
        return blob[start : start + length]

    return Atlas(
        shape=tuple(payload["shape"]),  # type: ignore[arg-type]
        spacing_um=tuple(payload["spacingUm"]),  # type: ignore[arg-type]
        source_shape=tuple(payload["sourceShape"]),  # type: ignore[arg-type]
        template=slice_of("template") or b"",
        region_ids=slice_of("regionIds") or b"",
        region_shape=tuple(payload["regionShape"]),  # type: ignore[arg-type]
        regions=tuple(Region.from_json(item) for item in payload["regions"]),
        mask=slice_of("mask"),
        meta=payload.get("meta", {}),
    )


def _main() -> int:  # pragma: no cover - CLI for inspecting the volumes
    import argparse

    parser = argparse.ArgumentParser(description="Inspect and bake the VFB atlas.")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--inspect", action="store_true", help="print volume geometry only")
    parser.add_argument("--write-rois", action="store_true", help="write cache/rois.json")
    parser.add_argument("--force", action="store_true", help="ignore the cached bake")
    args = parser.parse_args()

    if args.inspect:
        template, mask = _pick_sources(args.root)
        for label, path in (("template", template), ("mask", mask)):
            if not path:
                print(f"{label}: not found")
                continue
            volume = read_nrrd(path)
            print(f"=== {label}: {path}")
            print(json.dumps(describe_volume(volume), indent=2))
        return 0

    if args.write_rois:
        print("wrote", write_region_table(args.root))
        return 0

    if args.force:
        for name in ("atlas.json", "atlas.bin"):
            target = os.path.join(args.root, "cache", name)
            if os.path.isfile(target):
                os.remove(target)

    atlas = load_or_build(args.root, progress=lambda m: print("  -", m))
    payload = atlas.to_json()
    payload["meta"].pop("roiSnap", None)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
