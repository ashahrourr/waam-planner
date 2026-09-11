"""Slice a mesh into layers and fill each layer with deposition paths.

Wire-arc deposition is not FDM with a different material. A welding bead is
4-8 mm wide against a 0.4 mm plastic extrusion, so a slicer written for plastic
produces paths that are wrong by an order of magnitude: contours inset by the
wrong amount, infill spaced for a nozzle that does not exist, and corners
tighter than a molten bead can hold.

Everything here is parameterised by bead geometry instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Bead:
    """Geometry of a single deposited weld bead."""

    width: float = 0.006        # m, typical GMAW bead
    height: float = 0.002       # m, layer height
    overlap: float = 0.30       # fraction of width that adjacent beads share

    @property
    def stepover(self) -> float:
        """Centre-to-centre spacing of neighbouring beads.

        Beads are roughly parabolic in cross-section, so butting them edge to
        edge leaves valleys that compound over layers. Overlapping by ~30% of
        the width is the usual compromise between a flat top surface and
        wasting wire.
        """
        return self.width * (1.0 - self.overlap)


@dataclass
class Layer:
    z: float
    contours: list[np.ndarray]      # closed loops, (n,2)
    infill: list[np.ndarray]        # open polylines, (n,2)

    @property
    def paths(self) -> list[np.ndarray]:
        return self.contours + self.infill

    def length(self) -> float:
        total = 0.0
        for p in self.contours:
            total += float(np.sum(np.linalg.norm(np.diff(p, axis=0, append=p[:1]), axis=1)))
        for p in self.infill:
            total += float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        return total


def slice_heights(z_min: float, z_max: float, bead: Bead) -> np.ndarray:
    """Layer mid-heights, one bead tall each."""
    n = max(1, int(math.floor((z_max - z_min) / bead.height)))
    return z_min + bead.height * (np.arange(n) + 0.5)


def inset(loop: np.ndarray, distance: float) -> np.ndarray:
    """Shrink a convex-ish closed loop toward its centroid.

    A proper offset needs a straight-skeleton or clipper library; for the
    convex profiles used here, moving each vertex along its angle bisector is
    exact enough and keeps the dependency list empty.
    """
    centre = loop.mean(axis=0)
    out = []
    for point in loop:
        direction = point - centre
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            out.append(point)
            continue
        shrunk = norm - distance
        if shrunk <= 1e-6:
            return np.empty((0, 2))
        out.append(centre + direction / norm * shrunk)
    return np.array(out)


def zigzag(loop: np.ndarray, bead: Bead, angle: float = 0.0) -> list[np.ndarray]:
    """Fill a loop with alternating passes at `angle`.

    Consecutive passes run in opposite directions so the torch never has to lift
    and travel back — a restart means re-striking the arc, which leaves a defect.
    The fill direction is rotated layer to layer by the caller so the seams do
    not stack into a weak plane.
    """
    if len(loop) < 3:
        return []
    c, s = math.cos(-angle), math.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    centre = loop.mean(axis=0)
    local = (loop - centre) @ R.T

    y_min, y_max = local[:, 1].min(), local[:, 1].max()
    lines: list[np.ndarray] = []
    y = y_min + bead.stepover * 0.5
    flip = False
    while y < y_max:
        hits = _scanline(local, y)
        for i in range(0, len(hits) - 1, 2):
            a, b = hits[i], hits[i + 1]
            if b - a < bead.width * 0.5:        # too narrow for a bead
                continue
            seg = np.array([[a, y], [b, y]]) if not flip else np.array([[b, y], [a, y]])
            lines.append(seg @ np.linalg.inv(R).T + centre)
        flip = not flip
        y += bead.stepover
    return lines


def _scanline(poly: np.ndarray, y: float) -> list[float]:
    """x coordinates where the horizontal line at `y` crosses the polygon."""
    xs: list[float] = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 <= y < y2) or (y2 <= y < y1):
            t = (y - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    return sorted(xs)


def slice_prism(profile: np.ndarray, height: float, bead: Bead,
                rotate_fill: float = math.radians(90.0)) -> list[Layer]:
    """Slice an extruded 2-D profile — enough for the shapes WAAM is used for.

    Real WAAM parts are overwhelmingly walls, stiffeners, flanges and bosses:
    prismatic shapes where the cross-section is constant or changes slowly. A
    full mesh slicer is a different project; this covers the useful cases and
    keeps the focus on the path and kinematics side.
    """
    layers: list[Layer] = []
    for index, z in enumerate(slice_heights(0.0, height, bead)):
        wall = inset(profile, bead.width * 0.5)
        if len(wall) == 0:
            continue
        interior = inset(wall, bead.stepover)
        fill = zigzag(interior, bead, angle=index * rotate_fill) if len(interior) else []
        layers.append(Layer(z=float(z), contours=[wall], infill=fill))
    return layers


# ---- handy profiles ----------------------------------------------------
def rectangle(width: float, depth: float, corner: float = 0.0, points: int = 8) -> np.ndarray:
    w, d = width / 2.0, depth / 2.0
    if corner <= 0:
        return np.array([[-w, -d], [w, -d], [w, d], [-w, d]])
    loop = []
    for cx, cy, start in ((w - corner, d - corner, 0.0),
                          (-(w - corner), d - corner, math.pi / 2),
                          (-(w - corner), -(d - corner), math.pi),
                          (w - corner, -(d - corner), 3 * math.pi / 2)):
        for k in range(points):
            a = start + (math.pi / 2) * k / (points - 1)
            loop.append([cx + corner * math.cos(a), cy + corner * math.sin(a)])
    return np.array(loop)


def circle(radius: float, points: int = 48) -> np.ndarray:
    a = np.linspace(0.0, 2 * np.pi, points, endpoint=False)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])
