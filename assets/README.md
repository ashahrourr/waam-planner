# Vendored assets

## ur5e/

The Universal Robots UR5e model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), used so
the renders show the real robot rather than a stand-in built from primitives.

Licensed under the terms in [`ur5e/LICENSE`](ur5e/LICENSE) — BSD, Copyright 2018
ROS Industrial Consortium. Unmodified except for a scene file this repository
writes alongside it at render time.

The Denavit-Hartenberg parameters in `waam/arm.py` are the published UR5e ones,
so the planner solves for the same robot the renderer draws. Forward kinematics
from the two agree to about 1.5 mm at the tool flange; the residual is a frame
convention at the attachment site, not a difference in link geometry.
