"""Draw the arm, the deposited part and the planned trajectories."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

INK = "#101319"
GRID = "#272d36"
ARM = "#d8dee9"
JOINT = "#8fa3bf"
BEAD = "#ff8c42"
DONE = "#ff5b2e"
BASE = "#4db8ff"


def _style(ax) -> None:
    ax.set_facecolor(INK)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.06, 0.07, 0.09, 1.0))
        pane._axinfo["grid"].update(color=GRID, linewidth=0.5)
    ax.tick_params(colors="#7c8899", labelsize=7)
    ax.set_xlabel("x (m)", color="#7c8899", fontsize=8)
    ax.set_ylabel("y (m)", color="#7c8899", fontsize=8)
    ax.set_zlabel("z (m)", color="#7c8899", fontsize=8)


def _fit(ax, plan, arm) -> None:
    """Frame the arm and the part together, with equal scale on every axis.

    Matplotlib's 3-D axes do not honour set_aspect("equal"), so a cube has to be
    imposed by hand or the part comes out visibly skewed.
    """
    pts = [arm.link_points(t.q[0]) for t in plan.trajectories[:1]]
    pts += [arm.link_points(t.q[-1]) for t in plan.trajectories[-1:]]
    pts += [t.xyz for t in plan.trajectories]
    pts.append(np.zeros((1, 3)))
    cloud = np.vstack(pts)

    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    centre = (lo + hi) / 2.0
    half = max((hi - lo).max(), 0.25) / 2.0 * 1.15
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(max(0.0, centre[2] - half), centre[2] + half)


def _draw_arm(ax, points: np.ndarray):
    line, = ax.plot(points[:, 0], points[:, 1], points[:, 2],
                    color=ARM, linewidth=2.6, solid_capstyle="round", zorder=6)
    dots = ax.scatter(points[:-1, 0], points[:-1, 1], points[:-1, 2],
                      s=26, color=JOINT, depthshade=False, zorder=7)
    torch = ax.scatter([points[-1, 0]], [points[-1, 1]], [points[-1, 2]],
                       s=55, color=BEAD, depthshade=False, zorder=8)
    return line, dots, torch


def plot_plan(plan, arm, out: str, title: str = "Planned deposition") -> str:
    """Static view: every planned bead, with the arm at the final pose."""
    fig = plt.figure(figsize=(9, 7), facecolor=INK)
    ax = fig.add_subplot(111, projection="3d")
    _style(ax)

    for traj in plan.trajectories:
        ax.plot(traj.xyz[:, 0], traj.xyz[:, 1], traj.xyz[:, 2],
                color=BEAD, linewidth=1.1, alpha=0.85)

    if plan.trajectories:
        _draw_arm(ax, arm.link_points(plan.trajectories[-1].q[-1]))

    ax.scatter([0], [0], [0], s=90, marker="s", color=BASE,
               depthshade=False, label="arm base")
    _fit(ax, plan, arm)
    ax.view_init(elev=24, azim=-62)
    ax.set_title(title, color="#e6edf3", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=INK)
    plt.close(fig)
    return out


def animate(plan, arm, out: str, seconds: float = 16.0, fps: int = 20,
            title: str = "Robotic wire-arc deposition", spin: float = 0.0) -> str:
    """Animate the build: the arm moving while the part accumulates."""
    # Flatten every waypoint into one timeline, remembering which bead it is on.
    q_all, xyz_all, bead_id = [], [], []
    for i, traj in enumerate(plan.trajectories):
        q_all.append(traj.q)
        xyz_all.append(traj.xyz)
        bead_id.extend([i] * len(traj.q))
    if not q_all:
        raise ValueError("nothing to animate")
    q_all = np.vstack(q_all)
    xyz_all = np.vstack(xyz_all)
    bead_id = np.array(bead_id)

    frames = int(seconds * fps)
    idx = np.linspace(0, len(q_all) - 1, frames).astype(int)

    fig = plt.figure(figsize=(9, 7), facecolor=INK)
    ax = fig.add_subplot(111, projection="3d")
    _style(ax)
    _fit(ax, plan, arm)
    ax.set_title(title, color="#e6edf3", fontsize=12, pad=12)
    ax.scatter([0], [0], [0], s=90, marker="s", color=BASE, depthshade=False)

    laid, = ax.plot([], [], [], color=DONE, linewidth=1.0, alpha=0.75)
    current, = ax.plot([], [], [], color=BEAD, linewidth=2.0)
    arm_line, arm_dots, torch = _draw_arm(ax, arm.link_points(q_all[0]))
    hud = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color="#e6edf3",
                    fontsize=9, family="monospace")

    def update(f):
        i = idx[f]
        pts = arm.link_points(q_all[i])
        arm_line.set_data(pts[:, 0], pts[:, 1])
        arm_line.set_3d_properties(pts[:, 2])
        arm_dots._offsets3d = (pts[:-1, 0], pts[:-1, 1], pts[:-1, 2])
        torch._offsets3d = ([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]])

        here = bead_id[i]
        done = xyz_all[: i + 1][bead_id[: i + 1] < here]
        laid.set_data(done[:, 0], done[:, 1])
        laid.set_3d_properties(done[:, 2])
        live = xyz_all[: i + 1][bead_id[: i + 1] == here]
        current.set_data(live[:, 0], live[:, 1])
        current.set_3d_properties(live[:, 2])

        layer = plan.trajectories[here].layer
        hud.set_text(f"layer {layer + 1:2d}   bead {here + 1:3d}/{len(plan.trajectories)}"
                     f"   z {xyz_all[i, 2] * 1000:5.1f} mm")
        if spin:
            ax.view_init(elev=24, azim=-62 + spin * f / max(frames - 1, 1))
        return arm_line, arm_dots, torch, laid, current, hud

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    if out.endswith(".gif"):
        anim.save(out, writer=PillowWriter(fps=fps))
    else:
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)
    return out
