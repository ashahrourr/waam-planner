"""Draw the gantry machine: frame, rails, carriage, torch, part and arc.

Geometry comes from the Frame dimensions — extruded-aluminium uprights, a
cross-rail on the Y axis, a carriage on X, a Z post and a torch. Nothing here is
traced from a photograph.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

INK = "#0b0e13"
EXTRUSION = "#6c7889"
EXTRUSION_EDGE = "#49525f"
RAIL = "#aab6c6"
CARRIAGE = "#e3e9f2"
MOTOR = "#c4443a"
TORCH_BODY = "#8b95a5"
TORCH_TIP = "#ffc24d"
ARC = "#fff8dc"
BEAD = "#ff7a1a"
BED = "#171c25"
BED_EDGE = "#39424f"


def _box(cx, cy, cz, sx, sy, sz):
    """Six faces of an axis-aligned box centred at (cx, cy, cz)."""
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
             [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    return [c[f] for f in faces]


def _add_box(ax, cx, cy, cz, sx, sy, sz, color, edge=None, zorder=2, alpha=1.0):
    poly = Poly3DCollection(_box(cx, cy, cz, sx, sy, sz), facecolors=color,
                            edgecolors=edge or color, linewidths=0.4, alpha=alpha)
    poly.set_zorder(zorder)
    ax.add_collection3d(poly)
    return poly


def _style(ax, frame) -> None:
    ax.set_facecolor(INK)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.04, 0.05, 0.07, 1.0))
        pane._axinfo["grid"].update(color="#1b212b", linewidth=0.4)
    ax.tick_params(colors="#5d6878", labelsize=7)
    ax.set_xlabel("x (mm)", color="#5d6878", fontsize=8)
    ax.set_ylabel("y (mm)", color="#5d6878", fontsize=8)
    ax.set_zlabel("z (mm)", color="#5d6878", fontsize=8)
    mm = 1000.0
    ax.set_xlim(-20, frame.x * mm + 20)
    ax.set_ylim(-20, frame.y * mm + 20)
    ax.set_zlim(0, frame.z * mm + 60)
    ax.set_box_aspect((frame.x, frame.y, frame.z * 0.9))


def _draw_machine(ax, frame) -> None:
    """The fixed structure: bed, uprights, perimeter rails, motors.

    Drawn as thick lines rather than solid boxes. Matplotlib's 3-D backend
    composites artists with a painter's algorithm that does not depth-sort
    polygons against lines, so solid extrusions paint straight over the part
    sitting on the bed no matter what zorder they are given.
    """
    mm = 1000.0
    X, Y, Z = frame.x * mm, frame.y * mm, frame.z * mm

    def beam(p0, p1, width=5.0, color=EXTRUSION):
        ax.plot(*zip(p0, p1), color=color, linewidth=width,
                solid_capstyle="projecting", zorder=2)

    # bed outline and a light grid, so the plate reads as a surface
    for k in np.linspace(0, 1, 7):
        ax.plot([0, X], [k * Y, k * Y], [0, 0], color=BED_EDGE,
                linewidth=0.5, alpha=0.5, zorder=1)
        ax.plot([k * X, k * X], [0, Y], [0, 0], color=BED_EDGE,
                linewidth=0.5, alpha=0.5, zorder=1)

    for cx, cy in ((0, 0), (X, 0), (0, Y), (X, Y)):
        beam((cx, cy, 0), (cx, cy, Z), 6.0)

    for z in (0.0, Z):
        beam((0, 0, z), (X, 0, z), 5.0)
        beam((X, 0, z), (X, Y, z), 5.0)
        beam((X, Y, z), (0, Y, z), 5.0)
        beam((0, Y, z), (0, 0, z), 5.0)

    # stepper motors sit above everything, so solid boxes are safe here
    for cx, cy in ((0, 0), (X, Y)):
        _add_box(ax, cx, cy, Z + 22, 32, 32, 42, MOTOR, "#7d241d", zorder=6)


def _draw_head(ax, frame, pos_mm, arc_on: bool) -> list:
    """Cross-rail, carriage, Z post, torch and the arc at a tool position (mm)."""
    mm = 1000.0
    X, Z = frame.x * mm, frame.z * mm
    x, y, z = pos_mm
    rail_z = Z - 24.0
    out = []

    rail, = ax.plot([0, X], [y, y], [rail_z, rail_z], color=RAIL,
                    linewidth=4.5, solid_capstyle="projecting", zorder=7)
    carriage = ax.scatter([x], [y], [rail_z], s=150, marker="s",
                          color=CARRIAGE, depthshade=False, zorder=8)
    post, = ax.plot([x, x], [y, y], [rail_z, z + 46], color=CARRIAGE,
                    linewidth=3.5, zorder=8)
    body, = ax.plot([x, x], [y, y], [z + 46, z + 16], color=TORCH_BODY,
                    linewidth=7.0, solid_capstyle="round", zorder=9)
    tip, = ax.plot([x, x], [y, y], [z + 16, z + 3], color=TORCH_TIP,
                   linewidth=4.5, solid_capstyle="round", zorder=9)
    out += [rail, carriage, post, body, tip]

    out.append(ax.scatter([x], [y], [z], s=320 if arc_on else 0, color=ARC,
                          alpha=0.25, depthshade=False, zorder=10))
    out.append(ax.scatter([x], [y], [z], s=90 if arc_on else 20,
                          color=ARC if arc_on else "#7b8494",
                          depthshade=False, zorder=11))
    return out


def _tool_path(plan):
    """Tool positions (mm), arc state, and welded beads as polylines."""
    pts, lit, beads = [], [], []
    current: list[np.ndarray] = []
    here: np.ndarray | None = None
    arc = False
    for kind, value in plan.moves:
        if kind == "arc_on":
            arc = True
            # A straight infill pass emits one weld move; its other end is the
            # rapid that positioned the torch. Seed the bead with the current
            # position, or the whole pass is a single point and gets dropped.
            current = [here] if here is not None else []
        elif kind == "arc_off":
            arc = False
            if len(current) > 1:
                beads.append(np.array(current))
            current = []
        elif kind in ("rapid", "weld"):
            point = value * 1000.0
            pts.append(point)
            here = point
            welding = kind == "weld" and arc
            lit.append(welding)
            if welding:
                current.append(point)
    if len(current) > 1:
        beads.append(np.array(current))
    return np.array(pts), np.array(lit), beads


def plot_machine(plan, frame, out: str, title: str = "Gantry deposition") -> str:
    fig = plt.figure(figsize=(9.5, 7.5), facecolor=INK)
    ax = fig.add_subplot(111, projection="3d")
    _style(ax, frame)
    _draw_machine(ax, frame)

    pts, lit, beads = _tool_path(plan)
    # Drawn with plot() rather than a Line3DCollection: matplotlib composites
    # 3-D collections by their own depth ordering and paints the frame polygons
    # over a collection regardless of zorder, which hides the part entirely.
    for bead in beads:
        ax.plot(bead[:, 0], bead[:, 1], bead[:, 2],
                color=BEAD, linewidth=2.6, solid_capstyle="round")
    if len(pts):
        _draw_head(ax, frame, pts[-1], arc_on=False)

    ax.view_init(elev=22, azim=-52)
    ax.set_title(title, color="#e6edf3", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=INK)
    plt.close(fig)
    return out


def animate_machine(plan, frame, out: str, seconds: float = 16.0, fps: int = 20,
                    title: str = "Open-source metal printer") -> str:
    pts, lit, _ = _tool_path(plan)
    if not len(pts):
        raise ValueError("nothing to animate")

    frames = int(seconds * fps)
    idx = np.linspace(0, len(pts) - 1, frames).astype(int)

    fig = plt.figure(figsize=(9.5, 7.5), facecolor=INK)
    ax = fig.add_subplot(111, projection="3d")
    _style(ax, frame)
    _draw_machine(ax, frame)
    ax.view_init(elev=22, azim=-52)
    ax.set_title(title, color="#e6edf3", fontsize=12, pad=10)

    laid, = ax.plot([], [], [], color=BEAD, linewidth=2.6)
    hud = ax.text2D(0.02, 0.94, "", transform=ax.transAxes, color="#e6edf3",
                    fontsize=9.5, family="monospace")
    head: list = []

    def update(f):
        i = idx[f]
        # Only segments where the arc was lit at both ends; travel moves would
        # otherwise be drawn as beads straight across the part.
        # NaN breaks the polyline at travel moves, so rapids are not drawn as
        # beads straight across the part.
        trail = pts[: i + 1].astype(float).copy()
        trail[~lit[: i + 1]] = np.nan
        laid.set_data(trail[:, 0], trail[:, 1])
        laid.set_3d_properties(trail[:, 2])
        for artist in head:
            artist.remove()
        head[:] = _draw_head(ax, frame, pts[i], arc_on=bool(lit[i]))
        hud.set_text(f"{'ARC ON ' if lit[i] else 'travel '} "
                     f"x{pts[i, 0]:6.1f}  y{pts[i, 1]:6.1f}  z{pts[i, 2]:5.1f} mm")
        return (*head, laid, hud)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    if out.endswith(".gif"):
        anim.save(out, writer=PillowWriter(fps=fps))
    else:
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=3600))
    plt.close(fig)
    return out
