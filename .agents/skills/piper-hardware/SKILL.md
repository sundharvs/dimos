---
name: piper-hardware
description: Use when running, driving, or debugging the AgileX Piper arm on real hardware, e.g. bringing up CAN, starting piper-grasp, calling pick and place over RPC, checking what the wrist camera sees, or working out why the arm will not move.
---

# Piper on hardware

Scripts live in `scripts/` beside this file. Run them with the project interpreter (`.venv/bin/python`) from the repo root. `docs/capabilities/manipulation/piper_integration.md` is the user-facing reference; this skill is the working loop for an agent with no terminal UI and no eyes on the rig.

## Steps

1. **Run `scripts/preflight.py`.** It checks the CAN adapter, the arm, the cameras, the GPU, cached models and rig settings, and prints the fix for anything that fails. A missing `can0` needs the operator when sudo wants a password; your turn has to end before they can run it. Completion: no `FAIL` lines.
2. **Check every joint is healthy before blaming software.** `scripts/piper_joints.py` must show all six `enabled ... ok` once a stack is running. A joint that stays `DISABLED` with fault flags while the others are enabled makes the arm ignore joint moves while every trajectory still reports `COMPLETED`. Stop the stack, run `scripts/piper_joints.py --clear N`, start the stack again. All six reading `DISABLED ... collision, driver error` before any stack has connected is just the state after power-up; the adapter's connect sequence clears it.
3. **Start the stack with `scripts/stack.py start`** (`restart`, `stop`, `status`). It launches the blueprint detached, waits until it is ready or dead, prints why when it is dead, and clears what a killed stack leaves behind. Settings that belong to one rig go in `rig.ignore.env` in this folder (git-ignored `KEY=VALUE` lines): `PIPER_JUDGE_CAN=0` for a slcan adapter, the arm's `PIPER_JOINT_OFFSETS_DEG`, and `PIPER_SCENE_CAMERA`. The first start of a blueprint downloads several GB of model weights with no progress output.
4. **Drive it with `scripts/dimos_rpc.py`.** It is `dimos shell` without the terminal: a snippet on stdin gets `app`. The usual loop is `app.ManipulationSkills.go_home()`, `app.PickAndPlaceModule.scan_objects([...])`, `pick_object(object_id)`, `place_at(x, y)`.
5. **Look before and after every move.** `scripts/grab_frame.py out.png` saves what the wrist camera sees; read the image. Results that say `OK` are not evidence that the object is where it should be: re-scan and compare the object's cloud with the target. Measure with the camera facing the object (`move_to_joints` with the scan pose's joint 1 set to the object's azimuth). From the home pose an object off to one side is partly cut off by the frame and reads up to 1 cm from where it is.
6. **Park the arm when idle.** `app.ManipulationSkills.go_init()` returns it to the rest pose. Holding the scan pose loads the elbow and wrist motors and they warm up; `scripts/piper_joints.py` reports temperatures. `go_init` raises `Invalid goal configuration` when the arm started with a joint reading just past its limit (joint 2 rests on its lower stop); `dimos stop` parks the arm in that case.
7. **Stop with `scripts/stack.py stop`.** The adapter homes and disables the arm on the way out.

## Seeing the whole scene

`scripts/timelapse.py start` records the scene camera, one frame a second, and keeps `latest.jpg` current; `status` prints its path, `render` makes the video so far. Read `latest.jpg` for a third-person view of the arm and the object: the wrist camera cannot see the gripper touch anything, or an object taller than about 15 cm from the scan pose. The recorder owns the camera's colour stream, so do not open it elsewhere.

`scripts/line_up.py reference.png` lines the scene up with a reference wrist-camera image before a run: it blends the live wrist view with the reference and says how far to slide and turn the object, and whether the camera itself has moved. It opens the wrist camera directly, so stop the stack first, or pass a saved frame with `--image`.

`scripts/bottle_cap.py unscrew|screw x y top --watch x0,y0,x1,y1` works the cap of an upright, clamped bottle. It looks before it believes: read the contact sheet it prints the path of.

`piper-grasp-bin` picks up the yellow shelf bin by a wall: `app.ContainerPickModule.pick_up_container()`, then `set_down_container(x, y)` (and `rotate_held_container(deg)` in between to leave it in a new pose without the operator). Its verdict comes from the wrist camera; confirm with `latest.jpg`. If a pick reports failure, look before calling it again: the next call opens the jaws. A bin the skill cannot reach (no IK) can be turned by lowering the closed gripper beside one end and `move_linear` across it.

A tall object is also an obstacle the planner does not know. Before moving near it, register it with `app.ManipulationModule.add_obstacle(name, pose, "cylinder", [radius, height])` and retract upward before turning joint 1 past it.

## What goes wrong

- **Moves complete, nothing moves:** a faulted joint (step 2), or teaching mode (`arm: mode=teaching`, only the button on the base leaves it).
- **A pose target fails with `JOINT_LIMITS`:** the wrist pitch range bounds where the gripper can point straight down. `scripts/piper_reach.py 0.019 0.079` prints the reachable radius band at those tool heights.
- **A grasp lands off centre:** the object was cut off by the edge of the camera frame, or the cloud holds points from an earlier sighting. Print the cloud's extent before picking: a footprint shorter than the object means it is cut off (face it and scan again), a larger one means stale points.
- **The object ends up on its side:** nudge it over with the closed gripper (plan to a pose beside its top edge, `move_linear` across it) rather than waiting for the operator.
- **`Dimos.peek_stream` returns nothing for camera topics:** the camera is a native module, so use `scripts/grab_frame.py`.
