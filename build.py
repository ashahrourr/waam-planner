#!/usr/bin/env python3
"""Plan a wire-arc deposition build and render it.

    python build.py                                  # 90x60x12 mm block
    python build.py --shape tower --height 0.06
    python build.py --video docs/build.mp4 --plot docs/plan.png

Prints what was planned and, more usefully, what was refused and why.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from waam.arm import Arm
from waam.planner import plan
from waam.slicing import Bead, circle, rectangle, slice_prism

SHAPES = {
    "block":  lambda: rectangle(0.090, 0.060, corner=0.015),
    "tower":  lambda: rectangle(0.050, 0.050, corner=0.010),
    "ring":   lambda: circle(0.055),
    "plate":  lambda: rectangle(0.140, 0.090, corner=0.020),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shape", choices=sorted(SHAPES), default="block")
    ap.add_argument("--height", type=float, default=0.012, help="part height, m")
    ap.add_argument("--bead-width", type=float, default=0.006)
    ap.add_argument("--layer-height", type=float, default=0.002)
    ap.add_argument("--overlap", type=float, default=0.30)
    ap.add_argument("--origin", type=float, nargs=3, default=[-0.45, -0.15, 0.0],
                    help="where the work sits relative to the arm base, m")
    ap.add_argument("--plot", metavar="PATH")
    ap.add_argument("--video", metavar="PATH")
    ap.add_argument("--seconds", type=float, default=16.0)
    args = ap.parse_args(argv)

    arm = Arm()
    bead = Bead(width=args.bead_width, height=args.layer_height, overlap=args.overlap)
    profile = SHAPES[args.shape]()
    origin = np.array(args.origin, dtype=float)

    layers = slice_prism(profile, args.height, bead)
    print(f"shape   : {args.shape}, {args.height * 1000:.0f} mm tall")
    print(f"bead    : {bead.width * 1000:.1f} mm wide, {bead.height * 1000:.1f} mm layers, "
          f"{bead.stepover * 1000:.1f} mm stepover")
    print(f"sliced  : {len(layers)} layers\n")

    result = plan(layers, arm, bead, origin, profile)
    print(result.summary())

    if result.rejections:
        print("\nfirst few rejections:")
        for r in result.rejections[:5]:
            print(f"  layer {r.layer:2d} path {r.path:3d} point {r.point:3d}  "
                  f"{r.reason:9} {r.detail}")

    if not result.trajectories:
        print("\nnothing planned — move the work closer to the arm with --origin")
        return 1

    if args.plot:
        from waam.viz import plot_plan
        print("\nwrote", plot_plan(result, arm, args.plot,
                                   f"{args.shape} — {len(result.trajectories)} beads"))
    if args.video:
        from waam.viz import animate
        print("wrote", animate(result, arm, args.video, seconds=args.seconds,
                               title=f"{args.shape}: {result.bead_length:.1f} m of bead"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
