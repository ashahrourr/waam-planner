"""Six-axis arm kinematics.

The arm is described by Denavit-Hartenberg parameters, the standard way to
write down a serial manipulator: each joint contributes one 4x4 transform, and
multiplying them in order gives the pose of the tool relative to the base.

Geometry is loosely that of a UR5-class collaborative arm, which is the size
usually used for wire-arc deposition.

Forward kinematics is a product of matrices. Inverse kinematics — the joint
angles that put the torch at a requested pose — has no clean closed form for an
arbitrary target, so it is solved numerically with damped least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DH:
    """One joint: rotate about z by theta, then the fixed a/d/alpha offsets."""

    a: float        # link length, along x
    d: float        # link offset, along z
    alpha: float    # link twist, about x
    lower: float    # joint limits, radians
    upper: float


# UR5-like. Limits are tighter than the real robot's ±2π on purpose: a torch
# drags a welding cable and a gas hose, so unlimited wrist rotation is not free.
UR5 = (
    DH(0.0,     0.0892,  np.pi / 2, -2 * np.pi,  2 * np.pi),
    DH(-0.425,  0.0,     0.0,       -np.pi,      0.0),
    DH(-0.392,  0.0,     0.0,       -2.8,        2.8),
    DH(0.0,     0.1093,  np.pi / 2, -2 * np.pi,  2 * np.pi),
    DH(0.0,     0.0948, -np.pi / 2, -2 * np.pi,  2 * np.pi),
    DH(0.0,     0.0825,  0.0,       -3.0,        3.0),
)


def dh_transform(joint: DH, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(joint.alpha), np.sin(joint.alpha)
    return np.array([
        [ct, -st * ca,  st * sa, joint.a * ct],
        [st,  ct * ca, -ct * sa, joint.a * st],
        [0.0, sa,       ca,      joint.d],
        [0.0, 0.0,      0.0,     1.0],
    ])


class Arm:
    def __init__(self, joints: tuple[DH, ...] = UR5):
        self.joints = joints
        self.n = len(joints)
        # Upper bound on how far the tool can get from the base: every link
        # contributing its full length in the same direction. Loose, but it is
        # only used to reject targets that are obviously outside the workspace
        # before spending iterations on them.
        self.max_reach = sum(abs(j.a) + abs(j.d) for j in joints)

    @property
    def lower(self) -> np.ndarray:
        return np.array([j.lower for j in self.joints])

    @property
    def upper(self) -> np.ndarray:
        return np.array([j.upper for j in self.joints])

    # ---- forward --------------------------------------------------------
    def frames(self, q: np.ndarray) -> list[np.ndarray]:
        """Cumulative transform after each joint, base first."""
        out, T = [], np.eye(4)
        for joint, theta in zip(self.joints, q):
            T = T @ dh_transform(joint, float(theta))
            out.append(T.copy())
        return out

    def forward(self, q: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        for joint, theta in zip(self.joints, q):
            T = T @ dh_transform(joint, float(theta))
        return T

    def tool_position(self, q: np.ndarray) -> np.ndarray:
        return self.forward(q)[:3, 3]

    def link_points(self, q: np.ndarray) -> np.ndarray:
        """Joint origins including the base, for drawing and collision checks."""
        pts = [np.zeros(3)]
        pts.extend(T[:3, 3] for T in self.frames(q))
        return np.array(pts)

    # ---- differential ---------------------------------------------------
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Geometric Jacobian, 6xn: how tool twist responds to joint rates.

        Column i is the effect of joint i alone. For a revolute joint that is
        z_i x (p_tool - p_i) for the linear part and z_i for the angular part —
        the axis it turns about, crossed with the lever arm to the tool.
        """
        frames = self.frames(q)
        p_tool = frames[-1][:3, 3]
        J = np.zeros((6, self.n))
        z_prev, p_prev = np.array([0.0, 0.0, 1.0]), np.zeros(3)
        for i in range(self.n):
            J[:3, i] = np.cross(z_prev, p_tool - p_prev)
            J[3:, i] = z_prev
            z_prev = frames[i][:3, 2]
            p_prev = frames[i][:3, 3]
        return J

    # ---- inverse --------------------------------------------------------
    def inverse(self, target: np.ndarray, seed: np.ndarray | None = None,
                tol: float = 1e-4, iterations: int = 120,
                damping: float = 0.05, restarts: int = 12,
                rng: np.random.Generator | None = None) -> tuple[np.ndarray, bool]:
        """Joint angles placing the tool at `target` (position only).

        Damped least squares rather than a plain pseudo-inverse: near a
        singularity the Jacobian becomes ill-conditioned and an undamped step
        asks for enormous joint velocities. The damping term trades a little
        accuracy for a bounded step, which is what keeps the solver from
        launching the arm when a path grazes full extension.

        Seeding from the previous solution keeps successive points on a path in
        the same branch, so the arm does not flip configuration mid-bead.
        """
        # Outside the workspace there is nothing to converge to, and the solver
        # would burn every restart discovering that. Checking first turns a
        # hopeless target from ~0.1 s into a few microseconds, which matters
        # when a misplaced part means thousands of them.
        distance = float(np.linalg.norm(target))
        if distance > self.max_reach:
            return np.zeros(self.n), False

        rng = rng or np.random.default_rng(0)
        best_q, best_error = None, np.inf

        # Gradient descent on a non-convex problem lands in whatever basin the
        # seed falls in, and a six-axis arm has several configurations that
        # reach the same point. Retry from fresh seeds before giving up.
        for attempt in range(max(1, restarts)):
            if attempt == 0 and seed is not None:
                q = seed.astype(float).copy()
            elif attempt == 0:
                q = np.zeros(self.n)
            else:
                q = rng.uniform(self.lower * 0.8, self.upper * 0.8)

            for _ in range(iterations):
                error = target - self.tool_position(q)
                norm = np.linalg.norm(error)
                if norm < tol:
                    return self.wrap(q), True
                J = self.jacobian(q)[:3]
                # (J Jᵀ + λ²I)⁻¹ applied to the error, i.e. the damped step.
                JJt = J @ J.T + (damping ** 2) * np.eye(3)
                q = q + J.T @ np.linalg.solve(JJt, error)
                q = np.clip(q, self.lower, self.upper)

            final = np.linalg.norm(target - self.tool_position(q))
            if final < best_error:
                best_q, best_error = q.copy(), final

        return self.wrap(best_q), best_error < tol * 10

    @staticmethod
    def wrap(q: np.ndarray) -> np.ndarray:
        return (q + np.pi) % (2 * np.pi) - np.pi

    def within_limits(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self.lower - 1e-6) and np.all(q <= self.upper + 1e-6))
