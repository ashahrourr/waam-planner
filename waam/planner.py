"""Turn deposition paths into arm trajectories, refusing the ones that are unsafe.

Every point on every path is put through inverse kinematics, then checked
against three things that make a solution unusable even when the maths worked:

  reach       IK converged, but not to the requested point
  limits      a joint ended up outside its travel
  collision   a link passes through the part already deposited
  jump        the solution flipped to a different arm configuration mid-bead

The last one matters more than it sounds. A six-axis arm can reach the same
point several ways; if consecutive path points land in different branches the
arm swings through a large reconfiguration while the arc is lit, which ruins
the bead and can hit the work. Seeding each solve from the previous answer
usually prevents it, and this catches the cases where it does not.

Nothing is silently dropped. Rejections are collected with a reason, the same
way the drone's guard publishes its verdicts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .arm import Arm
from .slicing import Bead, Layer

MAX_JOINT_STEP = np.radians(25.0)      # per path point
CLEARANCE = 0.010                      # m, link to deposited metal


@dataclass
class Rejection:
    layer: int
    path: int
    point: int
    reason: str
    detail: str = ""


@dataclass
class Trajectory:
    """Joint-space trajectory for one continuous bead."""

    layer: int
    q: np.ndarray                  # (n, 6)
    xyz: np.ndarray                # (n, 3) tool positions actually achieved
    closed: bool = False


@dataclass
class Plan:
    trajectories: list[Trajectory] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    bead_length: float = 0.0

    def summary(self) -> str:
        points = sum(len(t.q) for t in self.trajectories)
        lines = [
            f"{len(self.trajectories)} beads, {points} waypoints, "
            f"{self.bead_length:.2f} m of deposition",
        ]
        if self.rejections:
            counts = Counter(r.reason for r in self.rejections)
            detail = ", ".join(f"{n}x {reason}" for reason, n in counts.most_common())
            lines.append(f"{len(self.rejections)} points rejected: {detail}")
        else:
            lines.append("no rejections")
        return "\n".join(lines)


class Part:
    """What has been deposited so far, for collision checking.

    Modelled as the extruded profile up to the height already built. That is
    conservative in the right direction — the real part is inside this — and
    cheap enough to check for every link of every waypoint.
    """

    def __init__(self, profile: np.ndarray, origin: np.ndarray):
        self.profile = profile
        self.origin = origin
        self.height = 0.0

    def grew_to(self, z: float) -> None:
        self.height = max(self.height, z)

    def contains(self, points: np.ndarray, clearance: float = 0.0) -> np.ndarray:
        """Which of `points` are inside the built solid, plus a clearance shell."""
        local = points[:, :2] - self.origin[:2]
        # Radial test against the profile: correct for the convex profiles used
        # here, and conservative (slightly large) for the rest.
        radius = np.linalg.norm(self.profile, axis=1).max() + clearance
        inside_xy = np.linalg.norm(local, axis=1) < radius
        below = points[:, 2] < self.origin[2] + self.height + clearance
        above_floor = points[:, 2] > self.origin[2] - clearance
        return inside_xy & below & above_floor


def resample(path: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    """Even points along a polyline, so waypoints are a fixed distance apart."""
    pts = np.vstack([path, path[:1]]) if closed else path
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    if total < 1e-9:
        return path[:1]
    n = max(2, int(np.ceil(total / spacing)) + 1)
    want = np.linspace(0.0, total, n)
    have = np.concatenate([[0.0], np.cumsum(seg)])
    return np.column_stack([np.interp(want, have, pts[:, i]) for i in range(pts.shape[1])])


def plan(layers: list[Layer], arm: Arm, bead: Bead, origin: np.ndarray,
         profile: np.ndarray, waypoint_spacing: float = 0.008) -> Plan:
    """Plan the whole build."""
    out = Plan()
    part = Part(profile, origin)
    seed: np.ndarray | None = None

    for li, layer in enumerate(layers):
        # Collisions are checked against what existed *before* this layer:
        # the bead being laid is at the tool, not an obstacle.
        part.grew_to(layer.z - bead.height)

        for pi, path2d in enumerate(layer.paths):
            closed = pi < len(layer.contours)
            pts2d = resample(path2d, waypoint_spacing, closed)
            world = np.column_stack([
                pts2d[:, 0] + origin[0],
                pts2d[:, 1] + origin[1],
                np.full(len(pts2d), origin[2] + layer.z),
            ])

            qs, achieved = [], []
            for wi, target in enumerate(world):
                q, ok = arm.inverse(target, seed=seed)
                reached = arm.tool_position(q)

                if not ok or np.linalg.norm(reached - target) > 1e-3:
                    out.rejections.append(Rejection(li, pi, wi, "reach",
                        f"{np.linalg.norm(reached - target) * 1000:.1f} mm short"))
                    continue
                if not arm.within_limits(q):
                    out.rejections.append(Rejection(li, pi, wi, "limits"))
                    continue
                links = arm.link_points(q)
                # Skip the last point: that *is* the torch tip, which is
                # supposed to be touching the work.
                if np.any(part.contains(links[:-1], CLEARANCE)):
                    out.rejections.append(Rejection(li, pi, wi, "collision"))
                    continue
                if qs and np.max(np.abs(q - qs[-1])) > MAX_JOINT_STEP:
                    out.rejections.append(Rejection(li, pi, wi, "jump",
                        f"{np.degrees(np.max(np.abs(q - qs[-1]))):.0f}deg"))
                    continue

                qs.append(q)
                achieved.append(reached)
                seed = q

            if len(qs) >= 2:
                q_arr = np.array(qs)
                xyz = np.array(achieved)
                out.trajectories.append(Trajectory(li, q_arr, xyz, closed))
                out.bead_length += float(
                    np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))

    return out
