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

from pathlib import Path

import numpy as np

EXTRUSION = "0.42 0.46 0.52 1"
RAIL = "0.68 0.72 0.78 1"
CARRIAGE = "0.86 0.89 0.94 1"
MOTOR = "0.72 0.24 0.20 1"
TORCH = "0.20 0.22 0.26 1"
TIP = "1.0 0.72 0.25 1"
BEAD_RGBA = (0.92, 0.38, 0.08)
HIDDEN = "0 0 0 0"

# Where the work actually sits on this machine. The planner works in part
# coordinates starting at z = 0; the bed's top face is 48 mm off the floor, so
# the render lifts the part onto it and drops the torch by the same amount.
BED_TOP = 0.048
NOZZLE_TIP = 0.055        # nozzle end in the gantry body frame, with z at zero


ROD = "0.72 0.75 0.80 1"
SCREW = "0.58 0.60 0.64 1"
PRINTED = "0.78 0.22 0.16 1"     # the red printed brackets on a RepRap
EXTRUSION = "0.40 0.43 0.48 1"
CARRIAGE = "0.82 0.85 0.90 1"
MOTOR = "0.17 0.18 0.21 1"
TORCH = "0.16 0.17 0.20 1"
TIP = "0.85 0.78 0.55 1"
BEAD_RGBA = (0.92, 0.38, 0.08)
HIDDEN = "0 0 0 0"

# Where the work actually sits on this machine. The planner works in part
# coordinates starting at z = 0; the bed's top face is 48 mm off the floor, so
# the render lifts the part onto it and drops the torch by the same amount.
BED_TOP = 0.048
NOZZLE_TIP = 0.055        # nozzle end in the gantry body frame, with z at zero


def _cyl(name, fromto, r, rgba):
    return (f'<geom name="{name}" type="cylinder" fromto="{fromto}" '
            f'size="{r}" rgba="{rgba}"/>')


def _box(name, pos, size, rgba):
    return f'<geom name="{name}" type="box" pos="{pos}" size="{size}" rgba="{rgba}"/>'


def build_xml(frame, segments: np.ndarray, bead_radius: float = 0.0032) -> str:
    """MJCF for a RepRap-style metal printer: welding torch on a moving gantry.

    Laid out like the machine it is modelled on — a Prusa-derived frame with
    printed brackets, smooth rods for the linear guides and lead screws for Z.

    The axes are split the way that frame actually splits them: **the bed carries
    the work and moves in Y**, while the torch moves in X and Z. So the part is a
    child of the bed body and travels with it. A planned point (px, py, pz)
    becomes x = px, z = pz, and a bed translation of Y/2 - py, which brings that
    point under a torch whose own Y never changes.
    """
    X, Y, Z = frame.x, frame.y, frame.z
    t = 0.010
    rod_y0, rod_y1 = 0.03, Y - 0.03
    gantry_h = Z - 0.05
    bed_y = Y / 2

    fixed: list[str] = []

    # base: extrusion rectangle on printed feet
    for i, (y) in enumerate((0.02, Y - 0.02)):
        fixed.append(_box(f"base{i}", f"{X/2} {y} {t}", f"{X/2} {t} {t}", EXTRUSION))
    for i, (cx, cy) in enumerate(((0.02, 0.02), (X - 0.02, 0.02),
                                  (0.02, Y - 0.02), (X - 0.02, Y - 0.02))):
        fixed.append(_box(f"foot{i}", f"{cx} {cy} {t/2}", "0.016 0.016 0.012", PRINTED))

    # two uprights, each a smooth rod plus a lead screw, joined by a top bar
    for i, x in enumerate((0.035, X - 0.035)):
        fixed.append(_cyl(f"zrod{i}", f"{x} {bed_y - 0.05} {t} {x} {bed_y - 0.05} {Z}",
                          0.005, ROD))
        fixed.append(_cyl(f"zscrew{i}", f"{x} {bed_y + 0.05} {t} {x} {bed_y + 0.05} {Z - 0.02}",
                          0.004, SCREW))
        fixed.append(_box(f"zmotor{i}", f"{x} {bed_y + 0.05} {t + 0.02}",
                          "0.021 0.021 0.021", MOTOR))
        fixed.append(_box(f"ztop{i}", f"{x} {bed_y} {Z}", "0.022 0.075 0.010", PRINTED))
    fixed.append(_box("topbar", f"{X/2} {bed_y} {Z}", f"{X/2} {t} {t}", EXTRUSION))

    # Y rails the bed rides on
    for i, x in enumerate((X/2 - 0.08, X/2 + 0.08)):
        fixed.append(_cyl(f"yrod{i}", f"{x} {rod_y0} {0.045} {x} {rod_y1} {0.045}",
                          0.005, ROD))
    fixed.append(_box("ymotor", f"{X/2} {0.015} {0.045}", "0.021 0.021 0.021", MOTOR))

    beads = "\n".join(
        f'      <geom name="bead{i}" type="capsule" '
        f'fromto="{a[0]:.5f} {a[1]:.5f} {a[2] + BED_TOP:.5f} '
        f'{b[0]:.5f} {b[1]:.5f} {b[2] + BED_TOP:.5f}" '
        f'size="{bead_radius}" rgba="{HIDDEN}"/>'
        for i, (a, b) in enumerate(segments))

    return f"""
<mujoco model="waam-gantry">
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.002"/>

  <visual>
    <global offwidth="1600" offheight="1200"/>
    <headlight ambient="0.38 0.38 0.41" diffuse="0.55 0.55 0.58" specular="0.2 0.2 0.2"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.02" zfar="30"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.05 0.06 0.09" rgb2="0.01 0.01 0.02" width="256" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.13 0.14 0.17"
             rgb2="0.10 0.11 0.14" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="plate" rgba="0.26 0.27 0.30 1" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="{X*0.4} {Y*0.3} {Z*2.4}" dir="0 0.2 -1" directional="true"
           diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light pos="{-X*0.8} {-Y*0.5} {Z*1.5}" dir="1 0.8 -1" diffuse="0.28 0.28 0.32"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor" pos="0 0 0"/>

{chr(10).join("    " + f for f in fixed)}

    <!-- The bed carries the work and travels in Y. -->
    <body name="bed" pos="0 0 0">
      <joint name="y" type="slide" axis="0 1 0" limited="false"/>
      <geom name="bedplate" type="box" pos="{X/2} {bed_y} 0.042"
            size="{X/2 - 0.05} 0.085 0.006" material="plate"/>
      <geom name="bedbracket" type="box" pos="{X/2} {bed_y} 0.033"
            size="0.05 0.05 0.006" rgba="{PRINTED}"/>
{beads}
    </body>

    <!-- The torch moves in X and Z; its Y is fixed. -->
    <body name="zgantry" pos="0 0 0">
      <joint name="z" type="slide" axis="0 0 1" limited="false"/>
      <geom name="xrod0" type="cylinder"
            fromto="0.035 {bed_y - 0.018} {gantry_h} {X - 0.035} {bed_y - 0.018} {gantry_h}"
            size="0.005" rgba="{ROD}"/>
      <geom name="xrod1" type="cylinder"
            fromto="0.035 {bed_y + 0.018} {gantry_h} {X - 0.035} {bed_y + 0.018} {gantry_h}"
            size="0.005" rgba="{ROD}"/>
      <geom name="xendl" type="box" pos="0.035 {bed_y} {gantry_h}"
            size="0.016 0.030 0.020" rgba="{PRINTED}"/>
      <geom name="xendr" type="box" pos="{X - 0.035} {bed_y} {gantry_h}"
            size="0.016 0.030 0.020" rgba="{PRINTED}"/>
      <body name="xcar" pos="0 0 0">
        <joint name="x" type="slide" axis="1 0 0" limited="false"/>
        <geom name="carriage" type="box" pos="0 {bed_y} {gantry_h}"
              size="0.026 0.026 0.018" rgba="{CARRIAGE}"/>
        <geom name="torchclamp" type="box" pos="0 {bed_y} {gantry_h - 0.026}"
              size="0.018 0.018 0.010" rgba="{PRINTED}"/>
        <geom name="torch" type="capsule"
              fromto="0 {bed_y} {gantry_h - 0.034} 0 {bed_y} 0.075"
              size="0.010" rgba="{TORCH}"/>
        <geom name="nozzle" type="capsule"
              fromto="0 {bed_y} 0.075 0 {bed_y} 0.055" size="0.005" rgba="{TIP}"/>
        <site name="arc" pos="0 {bed_y} {NOZZLE_TIP - 0.004}" size="0.007"
              rgba="1 0.95 0.75 0"/>
        <geom name="hose" type="capsule"
              fromto="0 {bed_y} {gantry_h - 0.030} 0.04 {bed_y + 0.05} {gantry_h + 0.02}"
              size="0.006" rgba="0.10 0.10 0.12 1"/>
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


# ---------------------------------------------------------------- arm ----
UR5E_DIR = Path(__file__).resolve().parents[1] / "assets" / "ur5e"


def build_arm_scene(segments: np.ndarray, origin: np.ndarray,
                    bead_radius: float = 0.0032) -> str:
    """Scene XML wrapping the UR5e from MuJoCo Menagerie.

    The robot is the published model with its real meshes, not a stand-in built
    from primitives, so the render shows the machine the planner is actually
    solving for. A torch is attached at the model's own attachment_site — the
    flange where a real tool bolts on.

    Written next to the model because MJCF resolves include and meshdir paths
    relative to the file it is loaded from.
    """
    beads = "\n".join(
        f'    <geom name="bead{i}" type="capsule" '
        f'fromto="{a[0]:.5f} {a[1]:.5f} {a[2]:.5f} {b[0]:.5f} {b[1]:.5f} {b[2]:.5f}" '
        f'size="{bead_radius}" rgba="{HIDDEN}"/>'
        for i, (a, b) in enumerate(segments))

    return f"""
<mujoco model="waam-arm-scene">
  <include file="ur5e.xml"/>

  <visual>
    <global offwidth="1600" offheight="1200" azimuth="140" elevation="-22"/>
    <headlight ambient="0.35 0.35 0.38" diffuse="0.55 0.55 0.58" specular="0.15 0.15 0.15"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.02" zfar="40"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.05 0.06 0.09" rgb2="0.01 0.01 0.02" width="256" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.13 0.14 0.17"
             rgb2="0.10 0.11 0.14" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="10 10" reflectance="0.1"/>
    <material name="plate" rgba="0.32 0.34 0.38 1" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0.4 0.4 2.0" dir="-0.2 -0.2 -1" directional="true"
           diffuse="0.55 0.55 0.55" castshadow="true"/>
    <light pos="-1.4 -1.0 1.4" dir="1 0.7 -1" diffuse="0.28 0.28 0.32"/>
    <!-- The robot bolts to a pedestal and the work sits on a table below it,
         so the arm reaches down onto the job the way it would in a cell. With
         both at the same height the arm has to lie flat to reach its own base
         plane, which is both an awkward pose and a bad look. -->
    <geom name="floor" type="plane" size="4 4 0.05" material="floor"
          pos="0 0 {origin[2] - 0.10}"/>
    <geom name="pedestal" type="cylinder" pos="0 0 {(origin[2] - 0.10) / 2}"
          size="0.11 {abs(origin[2] - 0.10) / 2}" rgba="0.20 0.22 0.26 1"/>
    <geom name="table" type="box" pos="{origin[0]} {origin[1]} {origin[2] - 0.012}"
          size="0.24 0.24 0.012" material="plate"/>

{beads}
  </worldbody>

  <!-- The torch bolts to the flange the real robot mounts tools on. -->
  <worldbody>
    <body name="torchmount" mocap="true" pos="0 0 -5"/>
  </worldbody>
</mujoco>
"""


def write_arm_scene(segments: np.ndarray, origin: np.ndarray) -> Path:
    """Materialise the scene beside the UR5e model so includes resolve."""
    path = UR5E_DIR / "_waam_scene.xml"
    path.write_text(build_arm_scene(segments, origin))
    return path
