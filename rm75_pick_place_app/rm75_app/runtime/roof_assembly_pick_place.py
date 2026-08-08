from __future__ import annotations

import sys

from rm75_app.launch import run_app_module


TINGZI_OBJECTS = [
    "tingzi_base",
    "tingzi_pillar_front_left",
    "tingzi_pillar_front_right",
    "tingzi_pillar_back_left",
    "tingzi_pillar_back_right",
]


DEFAULT_ARGS = [
    "--wrist-relation-refine-after-grasp",
    "--wrist-relation-apply-mode",
    "during_transport",
    "--wrist-relation-timeout-s",
    "18.0",
    "--wrist-relation-wait-before-place-s",
    "0.0",
    "--wrist-relation-wait-at-pre-place-s",
    "1.5",
    "--wrist-relation-refined-place-max-candidates",
    "12",
    "--wrist-relation-require-refined-place",
    "--wrist-relation-extra-arg",
    "--pose-backend gripper_plane --mask-refine-mode gripper --mask-max-refine-candidates 1 --gripper-plane-refine-method smooth --gripper-plane-opt-maxiter 30 --gripper-plane-opt-point-count 700 --sam3-max-masks-per-item 5 --sam3-component-mode all",
    "--auto-execute",
    "--cycle-object-names",
    *TINGZI_OBJECTS,
    "--cycle-order-targets",
    "--no-reselect-target-on-planning-failure",
    "--insert-place-partial-open-at-release",
    "--insert-place-partial-open-delta",
    "0.04",
    "--insert-place-full-open-after-clearance",
    "--fast-chain-allow-legacy-fallback",
    "--joint-search-validate-final-contact",
    "--insert-joint-search-max-final-contact-checks",
    "4",
    "--target-selection-order",
    "cycle",
    "--insert-vertical-axial-spin-deg",
    "0.0",
    "90.0",
    "180.0",
    "270.0",
    "--tabletop-place-yaw-variant-deg",
    "0.0",
    "90.0",
    "180.0",
    "270.0",
    "--topdown-grasp-yaw-variant-deg",
    "0.0",
    "90.0",
    "180.0",
    "270.0",
    "--topdown-tilt-toward-robot-deg",
    "12.0",
    "20.0",
    "--topdown-tilt-toward-robot-shift-m",
    "0.0",
    "--direct-grasp-object-axis-shifts-m",
    "0.0",
    "--direct-grasp-z-lifts-m",
    "0.0",
    "--direct-elongated-grasp-tilt-toward-robot-deg",
    "0.0",
    "12.0",
    "20.0",
    "--direct-elongated-grasp-tilt-toward-robot-shift-m",
    "0.0",
    "--min-grasp-tcp-z",
    "0.025",
    "--curobo-world-mesh-object-names",
    "tingzi_base",
    "--place-insert-target-collision-scale",
    "0.18",
    "--targeted-place-hover-extra-height-m",
    "0.0",
    "0.01",
    "--direct-release-approach-distances",
    "0.030",
    "0.045",
    "0.060",
    "--transport-hover-extra-heights-m",
    "0.0",
    "0.030",
    "--curobo-approach-metric-locked-axis-tol-m",
    "0.025",
    "--dry-run-motion-window-scale",
    "0.45",
    "--dry-run-motion-window-min-s",
    "0.25",
    "--dry-run-motion-window-max-s",
    "3.0",
]


def main(argv: list[str] | None = None) -> int:
    user_args = list(sys.argv[1:] if argv is None else argv)
    return run_app_module("rm75_app.runtime.direct_pre_place", [*DEFAULT_ARGS, *user_args])


if __name__ == "__main__":
    raise SystemExit(main())
