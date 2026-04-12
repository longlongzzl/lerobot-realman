#!/usr/bin/env python3
"""Clean pure pick-place entrypoint.

This wrapper intentionally exposes the existing FoundationPose -> grasp ->
fixed-goal place pipeline without any rearrangement / TSIP search layer.
"""

from rm75_jiaobang_pick_real_with_foundationpose import main


if __name__ == "__main__":
    main()
