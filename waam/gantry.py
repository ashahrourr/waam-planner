"""Cartesian gantry: a welding torch on three linear axes.

This is the machine the open-source metal printers are built on — a MIG torch
bolted to a RepRap-style frame, as in Anzalone, Zhang, Wijnen, Sanders and
Pearce, "A Low-Cost Open-Source Metal 3-D Printer" (IEEE Access, 2013).

A gantry needs no inverse kinematics: the axes *are* the coordinates. What it
does need is the envelope check the arm also needs — a commanded point outside
the frame is a crash, not a move — plus G-code, because that is what the
firmware actually consumes.

Welding G-code is not plastic G-code:

  * the arc is struck and extinguished, not extruded continuously
  * travel between beads must be at a safe Z, or the torch drags through the
    part it just laid
  * the wire feed and travel speed together set heat input, which is the
    variable that decides whether the part warps
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .slicing import Bead, Layer


@dataclass
class Frame:
    """Work envelope, in metres. Defaults are a 300 mm-class printer."""

    x: float = 0.300
    y: float = 0.300
    z: float = 0.250
    safe_z: float = 0.020        # travel height above the part

    def contains(self, point: np.ndarray) -> bool:
        return bool(0 <= point[0] <= self.x and 0 <= point[1] <= self.y
                    and 0 <= point[2] <= self.z)


@dataclass
class Weld:
    """Process parameters. These are what you actually tune on the machine."""

    travel_speed: float = 0.008      # m/s while depositing
    rapid_speed: float = 0.050       # m/s while moving between beads
    wire_feed: float = 2.5           # m/min
    arc_on_dwell: float = 0.4        # s, let the puddle establish
    interlayer_dwell: float = 20.0   # s, let the layer cool


@dataclass
class Rejection:
    layer: int
    path: int
    point: int
    reason: str
    detail: str = ""


@dataclass
class GantryPlan:
    moves: list[tuple[str, np.ndarray]] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    bead_length: float = 0.0
    beads: int = 0
    estimated_seconds: float = 0.0

    def summary(self) -> str:
        lines = [f"{self.beads} beads, {self.bead_length:.2f} m of deposition, "
                 f"{self.estimated_seconds / 60:.0f} min estimated"]
        if self.rejections:
            counts = Counter(r.reason for r in self.rejections)
            lines.append(f"{len(self.rejections)} points rejected: " +
                         ", ".join(f"{n}x {r}" for r, n in counts.most_common()))
        else:
            lines.append("no rejections")
        return "\n".join(lines)


def plan_gantry(layers: list[Layer], frame: Frame, bead: Bead, weld: Weld,
                origin: np.ndarray) -> GantryPlan:
    """Plan a build on a Cartesian machine."""
    out = GantryPlan()
    previous_layer = -1

    for li, layer in enumerate(layers):
        for pi, path2d in enumerate(layer.paths):
            closed = pi < len(layer.contours)
            pts = np.vstack([path2d, path2d[:1]]) if closed else path2d
            world = np.column_stack([
                pts[:, 0] + origin[0],
                pts[:, 1] + origin[1],
                np.full(len(pts), origin[2] + layer.z),
            ])

            usable = []
            for wi, point in enumerate(world):
                if not frame.contains(point):
                    out.rejections.append(Rejection(li, pi, wi, "envelope",
                        f"({point[0] * 1000:.0f},{point[1] * 1000:.0f},"
                        f"{point[2] * 1000:.0f}) mm"))
                    continue
                usable.append(point)

            if len(usable) < 2:
                continue
            usable = np.array(usable)

            if li != previous_layer:
                out.moves.append(("dwell", np.array([weld.interlayer_dwell])))
                previous_layer = li

            # Lift, travel, drop, strike the arc, run the bead, extinguish.
            out.moves.append(("rapid", np.array([usable[0][0], usable[0][1],
                                                 usable[0][2] + frame.safe_z])))
            out.moves.append(("rapid", usable[0]))
            out.moves.append(("arc_on", np.array([weld.arc_on_dwell])))
            for point in usable[1:]:
                out.moves.append(("weld", point))
            out.moves.append(("arc_off", np.array([])))

            length = float(np.sum(np.linalg.norm(np.diff(usable, axis=0), axis=1)))
            out.bead_length += length
            out.beads += 1
            out.estimated_seconds += (length / weld.travel_speed
                                      + weld.arc_on_dwell + 1.0)

    out.estimated_seconds += weld.interlayer_dwell * max(0, previous_layer + 1)
    return out


def to_gcode(plan: GantryPlan, weld: Weld) -> str:
    """Emit G-code. M3/M5 switch the arc, the way a spindle would."""
    mm = 1000.0
    feed = weld.travel_speed * 60 * mm        # mm/min
    rapid = weld.rapid_speed * 60 * mm

    lines = [
        "; wire-arc deposition",
        f"; travel {weld.travel_speed * 1000:.1f} mm/s, wire {weld.wire_feed:.1f} m/min",
        f"; {plan.beads} beads, {plan.bead_length:.2f} m, "
        f"~{plan.estimated_seconds / 60:.0f} min",
        "G21 ; mm",
        "G90 ; absolute",
        "G28 ; home",
    ]
    arc = False
    for kind, value in plan.moves:
        if kind == "rapid":
            lines.append(f"G0 X{value[0]*mm:.2f} Y{value[1]*mm:.2f} "
                         f"Z{value[2]*mm:.2f} F{rapid:.0f}")
        elif kind == "weld":
            lines.append(f"G1 X{value[0]*mm:.2f} Y{value[1]*mm:.2f} "
                         f"Z{value[2]*mm:.2f} F{feed:.0f}")
        elif kind == "arc_on":
            lines.append(f"M3 S{weld.wire_feed * 100:.0f} ; arc on, wire feed")
            lines.append(f"G4 P{value[0]:.1f} ; establish the puddle")
            arc = True
        elif kind == "arc_off":
            lines.append("M5 ; arc off")
            arc = False
        elif kind == "dwell":
            # Never idle with the arc lit.
            if arc:
                lines.append("M5 ; arc off")
                arc = False
            lines.append(f"G4 P{value[0]:.0f} ; interlayer cool")
    lines += ["M5 ; arc off", "G0 Z50.00 ; clear", "M2 ; end"]
    return "\n".join(lines) + "\n"
