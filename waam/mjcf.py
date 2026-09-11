"""Build the gantry as a MuJoCo model and render a deposition run.

MuJoCo is used as a renderer with real geometry rather than as a physics
experiment: the machine is a kinematic chain of three slide joints, driven
directly to the planned positions. That buys correct occlusion, shadows and
materials, none of which matplotlib's 3-D axes provide — it composites artists
with a painter's algorithm and paints solid frame members straight over the part
on the bed.

Deposited metal is pre-declared as one capsule per bead segment and revealed as
the torch reaches it. A MuJoCo model is fixed once compiled, so geometry cannot
be added mid-run; alpha can be, and that is enough to show a part accumulating.
"""

from __future__ import annotations

import numpy as np

EXTRUSION = "0.42 0.46 0.52 1"
RAIL = "0.68 0.72 0.78 1"
CARRIAGE = "0.86 0.89 0.94 1"
MOTOR = "0.72 0.24 0.20 1"
TORCH = "0.20 0.22 0.26 1"
TIP = "1.0 0.72 0.25 1"
BEAD_RGBA = (0.92, 0.38, 0.08)
HIDDEN = "0 0 0 0"


def _beam(name: str, fromto: str, size: float, rgba: str) -> str:
    return (f'<geom name="{name}" type="box" fromto="{fromto}" '
            f'size="{size} {size}" rgba="{rgba}"/>')


def build_xml(frame, segments: np.ndarray, bead_radius: float = 0.0032) -> str:
    """MJCF for the machine plus one capsule per bead segment.

    `segments` is (n, 2, 3) in metres: the endpoints of every welded move.
    """
    X, Y, Z = frame.x, frame.y, frame.z
    t = 0.010                     # half-size of a 20 mm extrusion
    rail_z = Z - 0.028

    parts: list[str] = []

    # fixed frame: four uprights, and a perimeter at the base and the top
    for i, (cx, cy) in enumerate(((0, 0), (X, 0), (0, Y), (X, Y))):
        parts.append(_beam(f"post{i}", f"{cx} {cy} 0 {cx} {cy} {Z}", t, EXTRUSION))
    for j, z in enumerate((t, Z)):
        parts += [
            _beam(f"rail{j}a", f"0 0 {z} {X} 0 {z}", t, EXTRUSION),
            _beam(f"rail{j}b", f"{X} 0 {z} {X} {Y} {z}", t, EXTRUSION),
            _beam(f"rail{j}c", f"{X} {Y} {z} 0 {Y} {z}", t, EXTRUSION),
            _beam(f"rail{j}d", f"0 {Y} {z} 0 0 {z}", t, EXTRUSION),
        ]
    for i, (cx, cy) in enumerate(((0, 0), (X, Y))):
        parts.append(f'<geom name="motor{i}" type="box" pos="{cx} {cy} {Z + 0.024}" '
                     f'size="0.018 0.018 0.022" rgba="{MOTOR}"/>')

    # the part being built, hidden until the torch gets there
    beads = []
    for i, (a, b) in enumerate(segments):
        if np.linalg.norm(b - a) < 1e-6:
            b = b + np.array([1e-5, 0.0, 0.0])
        beads.append(
            f'<geom name="bead{i}" type="capsule" '
            f'fromto="{a[0]:.5f} {a[1]:.5f} {a[2]:.5f} {b[0]:.5f} {b[1]:.5f} {b[2]:.5f}" '
            f'size="{bead_radius}" rgba="{HIDDEN}"/>')

    return f"""
<mujoco model="waam-gantry">
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.002"/>

  <visual>
    <!-- MuJoCo's offscreen framebuffer defaults to 640x480; asking the renderer
         for more than that fails unless it is declared here. -->
    <global offwidth="1600" offheight="1200"/>
    <headlight ambient="0.45 0.45 0.48" diffuse="0.55 0.55 0.58" specular="0.2 0.2 0.2"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.02" zfar="30"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.05 0.06 0.09" rgb2="0.01 0.01 0.02" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.12 0.13 0.16"
             rgb2="0.09 0.10 0.13" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.08"/>
    <material name="plate" rgba="0.30 0.32 0.36 1" reflectance="0.15"/>
  </asset>

  <worldbody>
    <light pos="{X * 0.5} {Y * 0.5} {Z * 2.2}" dir="0 0 -1" directional="true"
           diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light pos="{-X} {-Y} {Z * 1.6}" dir="1 1 -1" diffuse="0.3 0.3 0.35"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor" pos="0 0 -0.05"/>
    <geom name="bed" type="box" pos="{X / 2} {Y / 2} -0.008"
          size="{X / 2} {Y / 2} 0.008" material="plate"/>

    {chr(10).join("    " + p for p in parts)}

    <!-- deposited metal, revealed as the torch reaches each segment -->
    {chr(10).join("    " + b for b in beads)}

    <!-- Y carriage rides the frame, X carriage rides it, Z post hangs below. -->
    <body name="ycar" pos="0 0 0">
      <joint name="y" type="slide" axis="0 1 0" limited="false"/>
      {_beam("crossrail", f"0 0 {rail_z} {X} 0 {rail_z}", 0.007, RAIL)}
      <body name="xcar" pos="0 0 0">
        <joint name="x" type="slide" axis="1 0 0" limited="false"/>
        <geom name="carriage" type="box" pos="0 0 {rail_z}"
              size="0.024 0.018 0.016" rgba="{CARRIAGE}"/>
        <body name="zcar" pos="0 0 0">
          <joint name="z" type="slide" axis="0 0 1" limited="false"/>
          <geom name="zpost" type="capsule"
                fromto="0 0 {rail_z - 0.075} 0 0 0.052" size="0.008" rgba="{CARRIAGE}"/>
          <geom name="torch" type="capsule"
                fromto="0 0 0.050 0 0 0.018" size="0.011" rgba="{TORCH}"/>
          <geom name="tip" type="capsule"
                fromto="0 0 0.018 0 0 0.004" size="0.005" rgba="{TIP}"/>
          <site name="arc" pos="0 0 0.001" size="0.006" rgba="1 0.95 0.75 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def weld_segments(plan) -> np.ndarray:
    """Endpoints of every move made with the arc lit, in metres."""
    out, here, arc = [], None, False
    for kind, value in plan.moves:
        if kind == "arc_on":
            arc = True
        elif kind == "arc_off":
            arc = False
        elif kind in ("rapid", "weld"):
            if arc and kind == "weld" and here is not None:
                out.append((here, value))
            here = value
    return np.array(out) if out else np.zeros((0, 2, 3))


def tool_track(plan) -> tuple[np.ndarray, np.ndarray]:
    """Tool positions in metres and whether the arc was lit on arrival."""
    pts, lit, arc = [], [], False
    for kind, value in plan.moves:
        if kind == "arc_on":
            arc = True
        elif kind == "arc_off":
            arc = False
        elif kind in ("rapid", "weld"):
            pts.append(value)
            lit.append(arc and kind == "weld")
    return np.array(pts), np.array(lit)
