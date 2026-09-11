"""Draw the gantry machine: frame, rails, carriage, torch, and the part.

Geometry is the machine's own — a cubic frame of the configured envelope, an
X-rail riding the Y axis, a carriage on it, and a torch hanging below. Nothing
here is traced from a photograph; it is drawn from the Frame dimensions.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

INK = "#0e1116"
GRID = "#232932"
RAIL = "#8794a6"
FRAME = "#5a6675"
CARRIAGE = "#d8dee9"
TORCH = "#ffb020"
ARC = "#fff2c2"
BEAD = "#ff8c42"
DONE = "#e04a1f"
BED = "#1b2029"


def _style(ax, frame) -> None:
    ax.set_facecolor(INK)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.05, 0.06, 0.08, 1.0))
        pane._axinfo["grid"].update(color=GRID, linewidth=0.4)
    ax.tick_params(colors="#6f7b8c", labelsize=7)
    ax.set_xlabel("x (mm)", color="#6f7b8c", fontsize=8)
    ax.set_ylabel("y (mm)", color="#6f7b8c", fontsize=8)
    ax.set_zlabel("z (mm)", color="#6f7b8c", fontsize=8)
    mm = 1000.0
    ax.set_xlim(0, frame.x * mm)
    ax.set_ylim(0, frame.y * mm)
    ax.set_zlim(0, frame.z * mm * 0.75)
    ax.set_box_aspect((frame.x, frame.y, frame.z * 0.75))


def _draw_frame(ax, frame) -> None:
    """Four uprights and the top rails — the fixed structure."""
    mm = 1000.0
    X, Y, Z = frame.x * mm, frame.y * mm, frame.z * mm
    for cx, cy in ((0, 0), (X, 0), (0, Y), (X, Y)):
        ax.plot([cx, cx], [cy, cy], [0, Z], color=FRAME, linewidth=1.8, alpha=0.8)
    for z in (0.0, Z):
        ax.plot([0, X, X, 0, 0], [0, 0, Y, Y, 0], [z] * 5,
                color=FRAME, linewidth=1.4, alpha=0.75)
    # print bed
    bed = np.array([[0, 0], [X, 0], [X, Y], [0, Y]])
    ax.plot(np.append(bed[:, 0], bed[0, 0]), np.append(bed[:, 1], bed[0, 1]),
            np.zeros(5), color=RAIL, linewidth=2.0)


def _draw_head(ax, frame, pos_mm):
    """Y-rail, carriage and torch at a tool position (mm)."""
    mm = 1000.0
    X, Z = frame.x * mm, frame.z * mm
    x, y, z = pos_mm
    rail, = ax.plot([0, X], [y, y], [Z * 0.92, Z * 0.92],
                    color=RAIL, linewidth=2.4)
    post, = ax.plot([x, x], [y, y], [Z * 0.92, z + 28],
                    color=CARRIAGE, linewidth=2.0)
    body = ax.scatter([x], [y], [Z * 0.92], s=70, marker="s",
                      color=CARRIAGE, depthshade=False)
    nozzle, = ax.plot([x, x], [y, y], [z + 28, z + 4],
                      color=TORCH, linewidth=4.0, solid_capstyle="round")
    tip = ax.scatter([x], [y], [z], s=48, color=ARC, depthshade=False, zorder=9)
    return rail, post, body, nozzle, tip


def _tool_path(plan) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Flatten the plan into tool positions (mm), arc state, and welded beads.

    Beads are kept as polylines rather than loose points: a straight G1 move
    only carries its endpoints, so scattering them draws a dotted outline
    instead of a bead.
    """
    pts, lit, beads = [], [], []
    current: list[np.ndarray] = []
    here: np.ndarray | None = None
    arc = False
    for kind, value in plan.moves:
        if kind == "arc_on":
            arc = True
            # The bead starts where the torch already is. A straight infill
            # pass emits a single weld move — its other end is the rapid that
            # positioned the torch — so without this the whole bead is a lone
            # point and gets dropped.
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
    fig = plt.figure(figsize=(9, 7), facecolor=INK)
    ax = fig.add_subplot(111, projection="3d")
    _style(ax, frame)
    _draw_frame(ax, frame)

    pts, lit, beads = _tool_path(plan)
    for bead in beads:
        ax.plot(bead[:, 0], bead[:, 1], bead[:, 2],
                color=BEAD, linewidth=1.6, alpha=0.9, solid_capstyle="round")
    if len(pts):
        _draw_head(ax, frame, pts[-1])

    ax.view_init(elev=24, azim=-58)
    ax.set_title(title, color="#e6edf3", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=INK)
    plt.close(fig)
    return out


def animate_machine(plan, frame, out: str, seconds: float = 16.0, fps: int = 20,
                    title: str = "Open-source metal printer") -> str:
    pts, lit, _ = _tool_path(plan)
    if not len(pts):
        raise ValueError("nothing to animate")

    frames = int(seconds * fps)
    idx = np.linspace(0, len(pts) - 1, frames).astype(int)

    fig = plt.figure(figsize=(9, 7), facecolor=INK)
    ax = fig.add_subplot(111, projection="3d")
    _style(ax, frame)
    _draw_frame(ax, frame)
    ax.set_title(title, color="#e6edf3", fontsize=12, pad=12)

    laid, = ax.plot([], [], [], color=DONE, linewidth=1.3, alpha=0.85)
    head = _draw_head(ax, frame, pts[0])
    hud = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color="#e6edf3",
                    fontsize=9, family="monospace")

    def update(f):
        i = idx[f]
        # Break the trail at travel moves with NaN, so rapids are not drawn
        # as beads across the part.
        trail = pts[: i + 1].copy()
        trail[~lit[: i + 1]] = np.nan
        laid.set_data(trail[:, 0], trail[:, 1])
        laid.set_3d_properties(trail[:, 2])
        for artist in head:
            artist.remove()
        head_new = _draw_head(ax, frame, pts[i])
        head[:] = head_new
        state = "ARC ON " if lit[i] else "travel "
        hud.set_text(f"{state} x {pts[i,0]:6.1f}  y {pts[i,1]:6.1f}  "
                     f"z {pts[i,2]:5.1f} mm")
        return (*head, laid, hud)

    head = list(head)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    if out.endswith(".gif"):
        anim.save(out, writer=PillowWriter(fps=fps))
    else:
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)
    return out
