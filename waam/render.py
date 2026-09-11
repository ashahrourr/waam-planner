"""Render a deposition run with MuJoCo."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from .mjcf import (BED_TOP, BEAD_RGBA, NOZZLE_TIP, build_xml, tool_track,
                   weld_segments)

WIDTH, HEIGHT = 1280, 800


def _camera(model, frame, azimuth: float, elevation: float, distance: float):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [frame.x * 0.5, frame.y * 0.5, frame.z * 0.38]
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    return cam


def render(plan, frame, out: str, seconds: float = 14.0, fps: int = 30,
           spin: float = 40.0, width: int = WIDTH, height: int = HEIGHT,
           still: str | None = None) -> str:
    """Animate the build. Writes an mp4 (or a gif, via ffmpeg)."""
    segments = weld_segments(plan)
    pts, lit = tool_track(plan)
    if not len(pts):
        raise ValueError("nothing to render")

    model = mujoco.MjModel.from_xml_string(build_xml(frame, segments))
    data = mujoco.MjData(model)

    bead_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"bead{i}")
                for i in range(len(segments))]
    arc_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arc")
    qx = model.joint("x").qposadr[0]
    qy = model.joint("y").qposadr[0]
    qz = model.joint("z").qposadr[0]

    # Map each frame of video to a waypoint, and to how many beads exist by then.
    frames = int(seconds * fps)
    idx = np.linspace(0, len(pts) - 1, frames).astype(int)
    laid_by_point = np.cumsum(lit.astype(int)) - 1     # bead index at each point

    renderer = mujoco.Renderer(model, height=height, width=width)
    tmp = Path(tempfile.mkdtemp(prefix="waam-"))
    written = []

    for f, i in enumerate(idx):
        # The bed carries the work in Y, so reaching part-coordinate py means
        # translating the bed by Y/2 - py, not moving the torch.
        data.qpos[qx] = pts[i, 0]
        data.qpos[qy] = frame.y / 2 - pts[i, 1]
        data.qpos[qz] = pts[i, 2] + BED_TOP - NOZZLE_TIP
        mujoco.mj_forward(model, data)

        # Reveal every bead deposited up to this waypoint.
        upto = min(max(laid_by_point[i], -1) + 1, len(bead_ids))
        for gid in bead_ids[:upto]:
            model.geom_rgba[gid] = (*BEAD_RGBA, 1.0)
        for gid in bead_ids[upto:]:
            model.geom_rgba[gid] = (0, 0, 0, 0)
        if arc_site >= 0:          # a scene without an arc site still renders
            model.site_rgba[arc_site] = (1.0, 0.96, 0.78, 0.85 if lit[i] else 0.0)

        cam = _camera(model, frame,
                      azimuth=135 + spin * f / max(frames - 1, 1),
                      elevation=-20, distance=frame.x * 2.4)
        renderer.update_scene(data, camera=cam)
        path = tmp / f"f{f:05d}.png"
        _save_png(renderer.render(), path)
        written.append(path)

        if still and f == frames - 1:
            # Final frame with everything visible, as the static figure.
            for gid in bead_ids:
                model.geom_rgba[gid] = (*BEAD_RGBA, 1.0)
            model.site_rgba[arc_site] = (0, 0, 0, 0)
            renderer.update_scene(data, camera=_camera(
                model, frame, azimuth=138, elevation=-21, distance=frame.x * 2.5))
            _save_png(renderer.render(), Path(still))

    _encode(tmp, out, fps)
    for path in written:
        path.unlink(missing_ok=True)
    tmp.rmdir()
    return out


def still(plan, frame, out: str, azimuth: float = 142.0,
          width: int = WIDTH, height: int = HEIGHT) -> str:
    """One frame with the finished part and the torch parked above it."""
    segments = weld_segments(plan)
    pts, _ = tool_track(plan)
    model = mujoco.MjModel.from_xml_string(build_xml(frame, segments))
    data = mujoco.MjData(model)

    for i in range(len(segments)):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"bead{i}")
        model.geom_rgba[gid] = (*BEAD_RGBA, 1.0)

    # Park over the middle of the part rather than wherever the last bead
    # finished: an end position near a frame upright puts the torch directly
    # behind it, and the still then looks as though the head is missing.
    centre = segments.reshape(-1, 3).mean(axis=0) if len(segments) else np.zeros(3)
    top = segments.reshape(-1, 3)[:, 2].max() if len(segments) else 0.0
    data.qpos[model.joint("x").qposadr[0]] = centre[0]
    data.qpos[model.joint("y").qposadr[0]] = frame.y / 2 - centre[1]
    data.qpos[model.joint("z").qposadr[0]] = top + BED_TOP - NOZZLE_TIP + 0.03
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=_camera(
        model, frame, azimuth=azimuth, elevation=-24, distance=frame.x * 2.3))
    _save_png(renderer.render(), Path(out))
    return out


def _save_png(pixels: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(pixels).save(path)


def _encode(folder: Path, out: str, fps: int) -> None:
    if out.endswith(".gif"):
        palette = folder / "palette.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i",
                        str(folder / "f%05d.png"), "-vf",
                        "fps=16,scale=860:-2:flags=lanczos,"
                        "palettegen=max_colors=96:stats_mode=diff",
                        str(palette)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(folder / "f%05d.png"), "-i", str(palette),
                        "-lavfi", "fps=16,scale=860:-2:flags=lanczos[v];"
                                  "[v][1:v]paletteuse=dither=none:diff_mode=rectangle",
                        "-loop", "0", out], check=True)
        palette.unlink(missing_ok=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(folder / "f%05d.png"), "-c:v", "libx264",
                        "-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", out], check=True)


# ----------------------------------------------------------------- arm ---
def _arm_scene(plan, arm, origin):
    from .mjcf import write_arm_scene
    segments = []
    for traj in plan.trajectories:
        segments.extend(traj.xyz[i:i + 2] for i in range(len(traj.xyz) - 1))
    segments = np.array(segments) if segments else np.zeros((0, 2, 3))

    model = mujoco.MjModel.from_xml_path(str(write_arm_scene(segments, origin)))
    data = mujoco.MjData(model)
    qadr = [model.joint(n).qposadr[0] for n in
            ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")]
    bead_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"bead{i}")
                for i in range(len(segments))]
    return model, data, qadr, bead_ids, segments


def _arm_camera(model, origin, azimuth: float, distance: float):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [origin[0] * 0.45, origin[1] * 0.45, origin[2] * 0.5 + 0.12]
    cam.azimuth = azimuth
    cam.elevation = -26
    cam.distance = distance
    return cam


def render_arm(plan, arm, origin, out: str, seconds: float = 14.0, fps: int = 30,
               spin: float = 0.0, width: int = WIDTH, height: int = HEIGHT) -> str:
    """Animate the arm building the part."""
    model, data, qadr, bead_ids, segments = _arm_scene(plan, arm, origin)
    q_all = np.vstack([t.q for t in plan.trajectories])
    # Waypoint i completes bead segment i-1, offset by the beads already laid.
    laid = np.concatenate([np.arange(len(t.q)) + off for t, off in
                           zip(plan.trajectories,
                               np.cumsum([0] + [len(t.q) - 1
                                                for t in plan.trajectories[:-1]]))])

    frames = int(seconds * fps)
    idx = np.linspace(0, len(q_all) - 1, frames).astype(int)
    renderer = mujoco.Renderer(model, height=height, width=width)
    tmp = Path(tempfile.mkdtemp(prefix="waam-arm-"))

    for f, i in enumerate(idx):
        data.qpos[qadr] = q_all[i]
        mujoco.mj_forward(model, data)
        upto = min(int(laid[i]), len(bead_ids))
        for gid in bead_ids[:upto]:
            model.geom_rgba[gid] = (*BEAD_RGBA, 1.0)
        for gid in bead_ids[upto:]:
            model.geom_rgba[gid] = (0, 0, 0, 0)
        renderer.update_scene(data, camera=_arm_camera(
            model, origin, 152 + spin * f / max(frames - 1, 1), 1.28))
        _save_png(renderer.render(), tmp / f"f{f:05d}.png")

    _encode(tmp, out, fps)
    for path in tmp.glob("f*.png"):
        path.unlink()
    tmp.rmdir()
    return out


def still_arm(plan, arm, origin, out: str,
              width: int = WIDTH, height: int = HEIGHT) -> str:
    """One frame with the whole part deposited."""
    model, data, qadr, bead_ids, _ = _arm_scene(plan, arm, origin)
    for gid in bead_ids:
        model.geom_rgba[gid] = (*BEAD_RGBA, 1.0)
    data.qpos[qadr] = plan.trajectories[-1].q[-1]
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=_arm_camera(model, origin, 152, 1.24))
    _save_png(renderer.render(), Path(out))
    return out
