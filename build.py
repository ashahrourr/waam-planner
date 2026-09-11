#!/usr/bin/env python3
"""Plan a wire-arc deposition build and render it.

    python build.py                                  # 90x60x12 mm block
    python build.py --shape tower --height 0.06
    python build.py --video docs/build.mp4 --plot docs/plan.png

Prints what was planned and, more usefully, what was refused and why.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from waam.arm import Arm
from waam.gantry import Frame, Weld, plan_gantry, to_gcode
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
    ap.add_argument("--machine", choices=("gantry", "arm"), default="gantry",
                    help="gantry: welder on 3 linear axes, like the open-source "
                         "metal printers. arm: torch on a 6-axis robot.")
    ap.add_argument("--shape", choices=sorted(SHAPES), default="block")
    ap.add_argument("--height", type=float, default=0.012, help="part height, m")
    ap.add_argument("--bead-width", type=float, default=0.006)
    ap.add_argument("--layer-height", type=float, default=0.002)
    ap.add_argument("--overlap", type=float, default=0.30)
    ap.add_argument("--origin", type=float, nargs=3, default=None,
                    help="where the work sits, m (defaults suit each machine)")
    ap.add_argument("--gcode", metavar="PATH", help="write G-code (gantry only)")
    ap.add_argument("--plot", metavar="PATH")
    ap.add_argument("--video", metavar="PATH")
    ap.add_argument("--seconds", type=float, default=16.0)
    args = ap.parse_args(argv)

    bead = Bead(width=args.bead_width, height=args.layer_height, overlap=args.overlap)
    profile = SHAPES[args.shape]()
    layers = slice_prism(profile, args.height, bead)

    # The two machines want the work in different places: a gantry's origin is
    # the corner of its bed, an arm's is its own base.
    if args.origin is not None:
        origin = np.array(args.origin, dtype=float)
    else:
        origin = (np.array([0.150, 0.150, 0.0]) if args.machine == "gantry"
                  else np.array([-0.45, -0.15, 0.0]))

    print(f"machine : {args.machine}")
    print(f"shape   : {args.shape}, {args.height * 1000:.0f} mm tall")
    print(f"bead    : {bead.width * 1000:.1f} mm wide, {bead.height * 1000:.1f} mm layers, "
          f"{bead.stepover * 1000:.1f} mm stepover")
    print(f"sliced  : {len(layers)} layers\n")

    arm = Arm()
    frame, weld = Frame(), Weld()
    if args.machine == "gantry":
        result = plan_gantry(layers, frame, bead, weld, origin)
    else:
        result = plan(layers, arm, bead, origin, profile)
    print(result.summary())

    if result.rejections:
        print("\nfirst few rejections:")
        for r in result.rejections[:5]:
            print(f"  layer {r.layer:2d} path {r.path:3d} point {r.point:3d}  "
                  f"{r.reason:9} {r.detail}")

    if args.machine == "gantry":
        if not result.beads:
            print("\nnothing planned — move the work onto the bed with --origin")
            return 1
        if args.gcode:
            Path(args.gcode).write_text(to_gcode(result, weld))
            print(f"\nwrote {args.gcode} "
                  f"({len(to_gcode(result, weld).splitlines())} lines)")
        if args.plot:
            from waam.render import still as render_still
            print("wrote", render_still(result, frame, args.plot))
        if args.video:
            from waam.render import render
            print("wrote", render(result, frame, args.video, seconds=args.seconds))
        return 0

    if not result.trajectories:
        print("\nnothing planned — move the work closer to the arm with --origin")
        return 1
    if args.gcode:
        print("\n--gcode applies to the gantry; an arm takes joint angles.")
    if args.plot:
        from waam.render import still_arm
        print("\nwrote", still_arm(result, arm, origin, args.plot))
    if args.video:
        from waam.render import render_arm
        print("wrote", render_arm(result, arm, origin, args.video,
                                  seconds=args.seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
