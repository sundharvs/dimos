---
name: piper-hardware
description: Use when running, driving, or debugging the AgileX Piper arm on real hardware, e.g. bringing up CAN, starting piper-grasp, calling pick and place over RPC, checking what the wrist camera sees, or working out why the arm will not move.
---

# Piper on hardware

Scripts live in `scripts/` beside this file. Run them with the project interpreter (`.venv/bin/python`) from the repo root. `docs/capabilities/manipulation/piper_integration.md` is the user-facing reference; this skill is the working loop for an agent with no terminal UI and no eyes on the rig.

## Steps

1. **Check the rig is attached.** `lsusb` should list the CAN adapter and the RealSense, and `ip -br link show can0` should show the interface UP. A missing `can0` with a serial adapter (`/dev/ttyACM0`) needs `sudo slcand -o -c -s8 /dev/ttyACM0 can0 && sudo ip link set can0 up`; ask the operator if sudo needs a password. Completion: `scripts/piper_joints.py` prints six joints.
2. **Check every joint is healthy before blaming software.** `scripts/piper_joints.py` must show all six `enabled ... ok` once a stack is running. A joint that is `DISABLED` with fault flags makes the arm ignore joint moves while every trajectory still reports `COMPLETED`. Stop the stack, run `scripts/piper_joints.py --clear N`, start the stack again.
3. **Start the stack in the background.** `dimos --can-port can0 run piper-grasp --daemon > run.log 2>&1 < /dev/null &`, then wait for `DimOS running in background` in the log. Redirect to a file: the daemon keeps a pipe open and the command never returns. A slcan `can0` reports no bitrate, so set `PIPER_JUDGE_CAN=0`; set `PIPER_JOINT_OFFSETS_DEG` to the arm's offsets. The first start downloads several GB of model weights with no progress output, so a long silent start is a download, not a hang.
4. **Drive it with `scripts/dimos_rpc.py`.** It is `dimos shell` without the terminal: a snippet on stdin gets `app`. The usual loop is `app.ManipulationSkills.go_home()`, `app.PickAndPlaceModule.scan_objects([...])`, `pick_object(object_id)`, `place_at(x, y)`.
5. **Look before and after every move.** `scripts/grab_frame.py out.png` saves what the wrist camera sees; read the image. Results that say `OK` are not evidence that the object is where it should be: re-scan from the home pose and compare the object's cloud with the target.
6. **Park the arm when idle.** `app.ManipulationSkills.go_init()` returns it to the rest pose. Holding the scan pose loads the elbow and wrist motors and they warm up; `scripts/piper_joints.py` reports temperatures.
7. **Stop with `dimos stop`.** The adapter homes and disables the arm on the way out.

## What goes wrong

- **Moves complete, nothing moves:** a faulted joint (step 2), or teaching mode (`arm: mode=teaching`, only the button on the base leaves it).
- **A pose target fails with `JOINT_LIMITS`:** the wrist pitch range bounds where the gripper can point straight down. `scripts/piper_reach.py 0.019 0.079` prints the reachable radius band at those tool heights.
- **A grasp lands off centre:** the object was cut off by the edge of the camera frame, or the cloud holds points from an earlier sighting. Print the cloud's extent before picking; a footprint larger than the object is the giveaway.
- **The object ends up on its side:** nudge it over with the closed gripper (plan to a pose beside its top edge, `move_linear` across it) rather than waiting for the operator.
- **`Dimos.peek_stream` returns nothing for camera topics:** the camera is a native module, so use `scripts/grab_frame.py`.
