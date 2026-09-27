"""Dependency-free NRRD reader tuned for Virtual Fly Brain exports.

Only the subset of the NRRD specification that VFB actually emits is
implemented: 3-D scalar volumes, ``uint8``/``int8``/``uint16``/``int16``/
``uint32``/``int32``/``float``/``double`` sample types, and the ``raw``,
``gzip``, ``bzip2`` and ``ascii`` encodings.

Two deliberate performance choices make this fast without NumPy:

* The decompressed sample buffer is kept as a single ``bytes`` object and is
  never converted element-by-element. Slicing a ``bytes`` object with a stride
  (``buf[start:start + nx:fx]``) is a C-level operation, so *strided
  subsampling* of a 119-megavoxel volume costs roughly a second of pure Python.
* Element conversion (for non-8-bit types) therefore happens only for the
  voxels we keep, never for the whole volume.

NRRD stores the first axis fastest (Fortran order), so a voxel at
``(x, y, z)`` lives at offset ``x + nx * (y + ny * z)``.  Everything in this
module preserves that layout, which is also the layout WebGL 3-D textures
expect along their x axis.
"""

from __future__ import annotations

import bz2
import gzip
import os
import struct
from dataclasses import dataclass, field
from typing import Iterable

__all__ = ["NrrdError", "Volume", "read_nrrd", "read_header"]

_STRUCT_CODE = {
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "float": "f",
    "double": "d",
}

_ITEMSIZE = {name: struct.calcsize("<" + code) for name, code in _STRUCT_CODE.items()}


class NrrdError(RuntimeError):
    """Raised when a file is not an NRRD volume we can read."""


def read_header(path: str) -> dict:
    """Parse just the header of *path*, without inflating the sample data."""
    with open(path, "rb") as handle:
        if handle.read(4) not in (b"NRRD",):
            raise NrrdError(f"{path!r} is not an NRRD file (bad magic)")
        handle.readline()  # consume the rest of the magic line, incl. version
        fields: dict[str, str] = {}
        while True:
            line = handle.readline()
            if not line or line in (b"\n", b"\r\n"):
                break
            text = line.decode("latin-1").rstrip("\r\n")
            if not text or text.startswith("#"):
                continue
            key, _, value = text.partition(":")
            fields[key.strip().lower()] = value.strip()
        payload_offset = handle.tell()
    fields["_payload_offset"] = str(payload_offset)
    fields["_path"] = path
    return fields


def _parse_vector(text: str, count: int) -> list[float]:
    """Parse an NRRD bracketed/parenthesised list of numbers into floats."""
    cleaned = text.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
    values: list[float] = []
    for token in cleaned.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            values.append(0.0)
    if len(values) < count:
        values.extend([0.0] * (count - len(values)))
    return values[:count]


@dataclass
class Volume:
    """A 3-D scalar volume in Fortran (x-fastest) order."""

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    units: tuple[str, str, str]
    dtype: str
    buf: bytes
    attrs: dict = field(default_factory=dict)

    # -- derived geometry -------------------------------------------------
    @property
    def extent(self) -> tuple[float, float, float]:
        """Physical size of the volume in microns."""
        return tuple(n * s for n, s in zip(self.shape, self.spacing))  # type: ignore[return-value]

    @property
    def voxels(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    # -- sampling ---------------------------------------------------------
    def value_at(self, x: int, y: int, z: int) -> int:
        nx, ny, _ = self.shape
        index = x + nx * (y + ny * z)
        if self.dtype == "uint8":
            return self.buf[index]
        code = _STRUCT_CODE[self.dtype]
        return int(struct.unpack_from("<" + code, self.buf, index * _ITEMSIZE[self.dtype])[0])

    def histogram(self, limit_voxels: int | None = None) -> list[int]:
        """256-bin histogram, optionally over a strided subsample."""
        counts = [0] * 256
        if limit_voxels and self.voxels > limit_voxels:
            step = max(1, self.voxels // limit_voxels)
            for value in self.buf[::step]:
                counts[value] += 1
            return counts
        for value in self.buf:
            counts[value] += 1
        return counts

    def downsampled_u8(
        self,
        factor: tuple[int, int, int] = (4, 4, 4),
        offset: tuple[int, int, int] = (0, 0, 0),
        window: tuple[float, float] | None = None,
    ) -> tuple[bytes, tuple[int, int, int]]:
        """Strided subsample reduced to 8-bit, returned with its new shape.

        *factor* is the decimation per axis, *offset* the first voxel to keep.
        *window* is the ``(low, high)`` source range mapped onto 0..255; when
        omitted the source type's natural range is used.
        """
        nx, ny, nz = self.shape
        fx, fy, fz = (max(1, int(f)) for f in factor)
        ox = max(1, len(range(offset[0], nx, fx)))
        oy = max(1, len(range(offset[1], ny, fy)))
        oz = max(1, len(range(offset[2], nz, fz)))

        row_len = ox
        out = bytearray(ox * oy * oz)

        if self.dtype == "uint8":
            lo, hi = window if window is not None else (0.0, 255.0)
            scale = 255.0 / max(1e-9, hi - lo)
            identity = window is None
            for k in range(oz):
                z = offset[2] + k * fz
                plane = z * nx * ny
                for j in range(oy):
                    y = offset[1] + j * fy
                    start = plane + y * nx + offset[0]
                    chunk = self.buf[start : start + nx : fx]
                    base = (k * oy + j) * row_len
                    if identity:
                        out[base : base + len(chunk)] = chunk
                    else:
                        out[base : base + len(chunk)] = bytes(
                            min(255, max(0, int((v - lo) * scale))) for v in chunk
                        )
            return bytes(out), (ox, oy, oz)

        # Generic path: decode only the retained voxels.
        code = _STRUCT_CODE[self.dtype]
        size = _ITEMSIZE[self.dtype]
        full = float(2 ** (8 * size)) if code not in ("f", "d") else 1.0
        lo, hi = window if window is not None else (0.0, full)
        scale = 255.0 / max(1e-9, hi - lo)
        stride = fx * size
        unpack = struct.Struct("<" + code * row_len).unpack
        nx_bytes = nx * size
        for k in range(oz):
            z = offset[2] + k * fz
            plane = z * nx * ny * size
            row_base = (k * oy) * row_len
            for j in range(oy):
                y = offset[1] + j * fy
                row_start = plane + y * nx_bytes
                chunk = self.buf[row_start + offset[0] * size : row_start + nx_bytes : stride]
                if len(chunk) < row_len * size:
                    chunk = chunk + b"\0" * (row_len * size - len(chunk))
                values = unpack(chunk[: row_len * size])
                out[row_base + j * row_len : row_base + j * row_len + row_len] = bytes(
                    min(255, max(0, int((v - lo) * scale))) for v in values
                )
        return bytes(out), (ox, oy, oz)

    def bounding_box(self, threshold: int = 1, sample_step: int = 2) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Voxel-space bounding box of everything at or above *threshold*.

        Runs on a strided subsample for speed, then snaps the result out to the
        nearest multiple of *sample_step* so the reported box is conservative.
        """
        nx, ny, nz = self.shape
        lo = [nx, ny, nz]
        hi = [-1, -1, -1]
        for k in range(0, nz, sample_step):
            plane = k * nx * ny
            hit_plane = False
            for j in range(0, ny, sample_step):
                start = plane + j * nx
                row = self.buf[start : start + nx : sample_step]
                found = [i for i, v in enumerate(row) if v >= threshold]
                if not found:
                    continue
                hit_plane = True
                lo[1] = min(lo[1], j)
                hi[1] = max(hi[1], j)
                first = found[0] * sample_step
                last = found[-1] * sample_step
                if first < lo[0]:
                    lo[0] = first
                if last > hi[0]:
                    hi[0] = last
            if hit_plane:
                lo[2] = min(lo[2], k)
                hi[2] = max(hi[2], k)
        if hi[0] < 0:
            return (0, 0, 0), (nx - 1, ny - 1, nz - 1)
        limits = (nx, ny, nz)
        lo = [max(0, v - sample_step) for v in lo]
        hi = [min(limit - 1, v + sample_step) for v, limit in zip(hi, limits)]
        return tuple(lo), tuple(hi)  # type: ignore[return-value]


def read_nrrd(path: str) -> Volume:
    """Read a full NRRD volume into memory."""
    fields = read_header(path)
    raw_shape = [int(v) for v in fields.get("sizes", "").split()]
    if len(raw_shape) != 3:
        raise NrrdError(f"{path!r}: expected a 3-D volume, got sizes={raw_shape!r}")
    dtype = fields.get("type", "uint8")
    if dtype not in _STRUCT_CODE:
        raise NrrdError(f"{path!r}: unsupported sample type {dtype!r}")

    offset = int(fields["_payload_offset"])
    with open(path, "rb") as handle:
        handle.seek(offset)
        payload = handle.read()

    encoding = fields.get("encoding", "raw").lower()
    if encoding in ("gzip", "gz") or payload[:2] == b"\x1f\x8b":
        buf = gzip.decompress(payload)
    elif encoding in ("bzip2", "bz2") or payload[:3] == b"BZh":
        buf = bz2.decompress(payload)
    elif encoding == "raw":
        buf = payload
    elif encoding == "ascii":
        values = [int(float(v)) for v in payload.split()]
        buf = struct.pack("<" + _STRUCT_CODE[dtype] * len(values), *values)
    else:
        raise NrrdError(f"{path!r}: unsupported encoding {encoding!r}")

    if fields.get("endian", "little").lower() == "big" and dtype not in ("uint8", "int8"):
        # Swap in place so downstream code can always assume little-endian.
        size = _ITEMSIZE[dtype]
        code = _STRUCT_CODE[dtype]
        count = len(buf) // size
        values = struct.unpack(">" + code * count, buf[: count * size])
        buf = struct.pack("<" + code * count, *values)

    expected = raw_shape[0] * raw_shape[1] * raw_shape[2] * _ITEMSIZE[dtype]
    if len(buf) < expected:
        raise NrrdError(f"{path!r}: expected {expected} sample bytes, decoded {len(buf)}")

    # NRRD space directions give the physical step along each axis.
    directions = fields.get("space directions", "")
    spacing = [1.0, 1.0, 1.0]
    if directions:
        cleaned = directions.replace("none", "0,0,0")
        groups = [g for g in cleaned.replace("(", "|").replace(")", "|").split("|") if g.strip()]
        for axis, group in enumerate(groups[:3]):
            parts = _parse_vector(group, 3)
            norm = sum(p * p for p in parts) ** 0.5
            if norm > 0:
                spacing[axis] = norm
    if "spacings" in fields:
        spacing = _parse_vector(fields["spacings"], 3)
    if spacing == [1.0, 1.0, 1.0] and "spacing" in fields:
        value = float(fields["spacing"])
        spacing = [value, value, value]

    units_field = fields.get("space units", "microns microns microns")
    units = tuple(part.strip().strip('"') for part in units_field.split()[:3])
    while len(units) < 3:
        units = units + (units[-1] if units else "microns",)

    return Volume(
        shape=(raw_shape[0], raw_shape[1], raw_shape[2]),
        spacing=(spacing[0], spacing[1], spacing[2]),
        units=units,  # type: ignore[arg-type]
        dtype=dtype,
        buf=buf[:expected],
        attrs={
            "source": path,
            "encoding": encoding,
            "space": fields.get("space", ""),
            "space origin": fields.get("space origin", ""),
            "comment": fields.get("comment", ""),
        },
    )


def find_nrrd_files(roots: Iterable[str]) -> list[str]:
    """Recursively collect ``*.nrrd`` paths beneath *roots*, sorted by size."""
    import os

    found: list[str] = []
    for root in roots:
        if os.path.isfile(root) and root.lower().endswith(".nrrd"):
            found.append(root)
            continue
        for base, _dirs, files in os.walk(root):
            for name in files:
                if name.lower().endswith(".nrrd"):
                    found.append(os.path.join(base, name))
    found.sort(key=lambda p: (os.path.getsize(p), p))
    return found


def _self_test() -> int:  # pragma: no cover - manual smoke check
    """Round-trip a tiny synthetic volume through the reader."""
    import gzip as _gzip
    import os
    import tempfile

    nx, ny, nz = 8, 4, 3
    data = bytes((x * 10 + y) % 256 for z in range(nz) for y in range(ny) for x in range(nx))
    header = (
        "NRRD0005\n"
        "type: uint8\n"
        "dimension: 3\n"
        "space dimension: 3\n"
        f"sizes: {nx} {ny} {nz}\n"
        "space directions: (0.5,0,0) (0,0.5,0) (0,0,1)\n"
        "encoding: gzip\n"
        'space units: "microns" "microns" "microns"\n'
        "\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False) as handle:
        handle.write(header.encode("latin-1") + _gzip.compress(data))
        path = handle.name
    try:
        volume = read_nrrd(path)
        assert volume.shape == (nx, ny, nz), volume.shape
        assert abs(volume.spacing[0] - 0.5) < 1e-9, volume.spacing
        assert volume.value_at(3, 2, 1) == data[3 + nx * (2 + ny * 1)]
        reduced, shape = volume.downsampled_u8((2, 2, 2))
        assert shape == (4, 2, 2), shape
        # x in {0,2,4,6}, y in {0,2}, z in {0,2} -> (10x + y) with x even
        expected = bytearray()
        for _z in (0, 2):
            for y in (0, 2):
                for x in (0, 2, 4, 6):
                    expected.append((10 * x + y) % 256)
        assert reduced == bytes(expected), (reduced, bytes(expected))
        assert volume.downsampled_u8((2, 2, 2), offset=(1, 1, 1))[1] == (4, 2, 1), "offset shape"
        lo, hi = volume.bounding_box(1, 1)
        assert lo == (0, 0, 0) and hi == (nx - 1, ny - 1, nz - 1), (lo, hi)
        print("nrrd self-test OK", volume.shape, volume.extent, shape)
        return 0
    finally:
        os.unlink(path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_self_test())
