"""Tests for kinematics, slicing and the planner's refusals.

Kinematics is checked against properties that must hold for any correct
implementation — round-tripping FK through IK, the Jacobian matching finite
differences — rather than against recorded numbers.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from waam.arm import Arm, DH, dh_transform                          # noqa: E402
from waam.planner import Part, plan, resample                       # noqa: E402
from waam.slicing import (Bead, circle, inset, rectangle,           # noqa: E402
                          slice_heights, slice_prism, zigzag)


@pytest.fixture
def arm() -> Arm:
    return Arm()


def reachable(arm: Arm, rng, n: int = 1) -> np.ndarray:
    """Joint vectors inside the limits, so the poses they produce are reachable."""
    lo = np.maximum(arm.lower, -2.5)
    hi = np.minimum(arm.upper, 2.5)
    return rng.uniform(lo, hi, size=(n, arm.n))


# ------------------------------------------------------------- kinematics
def test_dh_transform_is_rigid():
    """Every joint transform must be a rotation plus a translation."""
    T = dh_transform(DH(0.3, 0.1, math.pi / 3, -1, 1), 0.7)
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_forward_kinematics_is_deterministic(arm):
    q = np.array([0.2, -0.9, 0.6, 0.1, -0.3, 0.4])
    assert np.allclose(arm.tool_position(q), arm.tool_position(q))


def test_link_points_start_at_the_base(arm):
    pts = arm.link_points(np.zeros(arm.n))
    assert np.allclose(pts[0], np.zeros(3))
    assert len(pts) == arm.n + 1


def test_jacobian_matches_finite_differences(arm):
    """The analytic Jacobian must agree with numerically perturbing each joint."""
    q = np.array([0.3, -1.0, 0.8, -0.2, 0.5, 0.1])
    J = arm.jacobian(q)[:3]
    eps = 1e-6
    for i in range(arm.n):
        dq = np.zeros(arm.n)
        dq[i] = eps
        numeric = (arm.tool_position(q + dq) - arm.tool_position(q - dq)) / (2 * eps)
        assert np.allclose(J[:, i], numeric, atol=1e-5), f"joint {i}"


def test_inverse_round_trips_on_reachable_targets(arm):
    """FK then IK must return to the same tool position.

    Targets are sampled inside the joint limits. Sampling outside them produces
    poses the arm genuinely cannot reach, and the solver is right to fail.
    """
    rng = np.random.default_rng(0)
    errors = []
    for q_true in reachable(arm, rng, 60):
        target = arm.tool_position(q_true)
        q, ok = arm.inverse(target)
        assert ok
        errors.append(np.linalg.norm(target - arm.tool_position(q)))
    assert max(errors) < 1e-3


def test_inverse_respects_joint_limits(arm):
    rng = np.random.default_rng(1)
    for q_true in reachable(arm, rng, 20):
        q, _ = arm.inverse(arm.tool_position(q_true))
        assert arm.within_limits(q)


def test_unreachable_target_is_reported(arm):
    q, ok = arm.inverse(np.array([5.0, 5.0, 5.0]))   # far outside the workspace
    assert not ok


def test_seeding_keeps_nearby_targets_in_one_branch(arm):
    """Consecutive path points should not flip the arm to another configuration."""
    base = np.array([-0.45, -0.10, 0.10])
    q_prev, _ = arm.inverse(base)
    for step in np.linspace(0.002, 0.02, 8):
        q, ok = arm.inverse(base + np.array([step, 0.0, 0.0]), seed=q_prev)
        assert ok
        assert np.max(np.abs(q - q_prev)) < math.radians(20)
        q_prev = q


# ---------------------------------------------------------------- slicing
def test_stepover_shrinks_with_overlap():
    assert Bead(width=0.006, overlap=0.0).stepover == pytest.approx(0.006)
    assert Bead(width=0.006, overlap=0.5).stepover == pytest.approx(0.003)


def test_layer_count_follows_bead_height():
    bead = Bead(height=0.002)
    assert len(slice_heights(0.0, 0.010, bead)) == 5


def test_inset_shrinks_a_loop():
    loop = rectangle(0.10, 0.10)
    smaller = inset(loop, 0.01)
    span_before = loop[:, 0].max() - loop[:, 0].min()
    span_after = smaller[:, 0].max() - smaller[:, 0].min()
    assert span_after < span_before


def test_inset_past_the_centre_returns_nothing():
    assert len(inset(circle(0.01), 0.05)) == 0


def test_zigzag_alternates_direction():
    """Passes must reverse, so the torch never lifts between them."""
    lines = zigzag(rectangle(0.08, 0.08), Bead(), angle=0.0)
    assert len(lines) > 2
    first_dx = lines[0][1, 0] - lines[0][0, 0]
    second_dx = lines[1][1, 0] - lines[1][0, 0]
    assert first_dx * second_dx < 0


def test_slice_prism_produces_contours_and_fill():
    layers = slice_prism(rectangle(0.09, 0.06, corner=0.015), 0.010, Bead())
    assert len(layers) == 5
    assert all(layer.contours for layer in layers)
    assert sum(len(layer.infill) for layer in layers) > 0
    assert layers[0].z < layers[-1].z


def test_fill_direction_rotates_between_layers():
    """Stacking every layer's seams in one plane makes a weak part."""
    layers = slice_prism(rectangle(0.09, 0.09), 0.006, Bead())
    def angle_of(layer):
        seg = layer.infill[0]
        return math.atan2(seg[1, 1] - seg[0, 1], seg[1, 0] - seg[0, 0]) % math.pi
    assert abs(angle_of(layers[0]) - angle_of(layers[1])) > 0.5


# ---------------------------------------------------------------- planner
def test_resample_gives_even_spacing():
    path = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]])
    pts = resample(path, 0.01, closed=False)
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert gaps.max() - gaps.min() < 1e-6


def test_part_contains_only_what_was_built():
    part = Part(rectangle(0.10, 0.10), np.zeros(3))
    part.grew_to(0.02)
    inside = np.array([[0.0, 0.0, 0.01]])
    above = np.array([[0.0, 0.0, 0.30]])
    assert part.contains(inside)[0]
    assert not part.contains(above)[0]


def test_plans_a_reachable_part(arm):
    bead = Bead()
    profile = rectangle(0.09, 0.06, corner=0.015)
    result = plan(slice_prism(profile, 0.008, bead), arm, bead,
                  np.array([-0.45, -0.15, 0.0]), profile)
    assert result.trajectories
    assert result.bead_length > 1.0
    assert not result.rejections


def test_out_of_reach_part_is_refused_entirely(arm):
    bead = Bead()
    profile = rectangle(0.09, 0.06, corner=0.015)
    result = plan(slice_prism(profile, 0.006, bead), arm, bead,
                  np.array([-1.60, -0.15, 0.0]), profile)
    assert not result.trajectories
    assert all(r.reason == "reach" for r in result.rejections)


def test_tall_part_triggers_collision_refusals(arm):
    """Once the tower is high enough, the arm cannot stay clear of it.

    The threshold moved when the kinematics changed from UR5 to the real UR5e:
    its base is 162 mm rather than 89 mm, so the arm clears a taller part before
    its links start intersecting what has been deposited.
    """
    bead = Bead()
    profile = rectangle(0.05, 0.05, corner=0.01)
    result = plan(slice_prism(profile, 0.28, bead), arm, bead,
                  np.array([-0.40, -0.10, 0.0]), profile)
    assert any(r.reason == "collision" for r in result.rejections)


def test_short_part_has_clearance(arm):
    """The same tower well under that threshold must plan without refusals."""
    bead = Bead()
    profile = rectangle(0.05, 0.05, corner=0.01)
    result = plan(slice_prism(profile, 0.13, bead), arm, bead,
                  np.array([-0.40, -0.10, 0.0]), profile)
    assert result.trajectories
    assert not result.rejections


def test_trajectories_are_continuous(arm):
    """No planned bead may contain a configuration flip."""
    bead = Bead()
    profile = rectangle(0.09, 0.06, corner=0.015)
    result = plan(slice_prism(profile, 0.008, bead), arm, bead,
                  np.array([-0.45, -0.15, 0.0]), profile)
    for traj in result.trajectories:
        steps = np.abs(np.diff(traj.q, axis=0))
        assert steps.max() <= math.radians(25) + 1e-9


def test_planned_points_land_where_asked(arm):
    bead = Bead()
    profile = rectangle(0.09, 0.06, corner=0.015)
    result = plan(slice_prism(profile, 0.006, bead), arm, bead,
                  np.array([-0.45, -0.15, 0.0]), profile)
    for traj in result.trajectories[:5]:
        for q, xyz in zip(traj.q, traj.xyz):
            assert np.linalg.norm(arm.tool_position(q) - xyz) < 1e-6


# ----------------------------------------------------------------- gantry
from waam.gantry import Frame, Weld, plan_gantry, to_gcode          # noqa: E402


def gantry_plan(height=0.008, origin=(0.150, 0.150, 0.0), frame=None):
    bead = Bead()
    profile = rectangle(0.09, 0.06, corner=0.015)
    return plan_gantry(slice_prism(profile, height, bead),
                       frame or Frame(), bead, Weld(), np.array(origin)), bead


def test_gantry_plans_a_part_on_the_bed():
    result, _ = gantry_plan()
    assert result.beads > 0
    assert result.bead_length > 0.5
    assert not result.rejections


def test_gantry_refuses_points_off_the_bed():
    result, _ = gantry_plan(origin=(0.290, 0.290, 0.0))
    assert any(r.reason == "envelope" for r in result.rejections)


def test_frame_envelope_bounds():
    frame = Frame(x=0.3, y=0.3, z=0.25)
    assert frame.contains(np.array([0.15, 0.15, 0.10]))
    assert not frame.contains(np.array([0.35, 0.15, 0.10]))
    assert not frame.contains(np.array([0.15, 0.15, -0.01]))


def test_gcode_is_well_formed():
    result, _ = gantry_plan()
    text = to_gcode(result, Weld())
    lines = text.splitlines()
    assert lines[0].startswith(";")
    assert "G21" in text and "G90" in text        # mm, absolute
    assert text.rstrip().endswith("M2 ; end")
    assert text.count("M3") == text.count("M5") - 1   # one extra M5 at the end


def test_arc_is_never_left_on_during_a_dwell():
    """An interlayer pause with the arc lit burns a hole in the part."""
    result, _ = gantry_plan(height=0.006)
    lines = to_gcode(result, Weld()).splitlines()
    arc = False
    for line in lines:
        if line.startswith("M3"):
            arc = True
        elif line.startswith("M5"):
            arc = False
        elif line.startswith("G4 P2"):            # the interlayer cool
            assert not arc


def test_every_weld_move_is_inside_the_envelope():
    frame = Frame()
    result, _ = gantry_plan(frame=frame)
    for kind, value in result.moves:
        if kind in ("rapid", "weld"):
            assert frame.contains(value)


def test_travel_between_beads_lifts_clear():
    """Rapids must go up before crossing, or the torch drags through the part."""
    result, _ = gantry_plan()
    rapids = [v for k, v in result.moves if k == "rapid"]
    assert len(rapids) >= 2
    # Rapids come in pairs: up to safe height, then down onto the start.
    assert rapids[0][2] > rapids[1][2]
