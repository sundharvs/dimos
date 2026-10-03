# Porting the container pick to the AgileX Piper

A guide for the agent that will run the container-pick auto-research on a
different arm. It records what the xArm7 run needed, what cost time, and what to
do in which order, so the Piper run is a few hours rather than a day. Read
[the container pick page](/docs/capabilities/manipulation/container-pick.md) first for what the skill does.

## 0. Before touching the arm (30 min)

1. **Read the robot's package**: `dimos/robot/manipulators/piper/config.py` and
   `blueprints/scene.py`. Facts that matter for the port:
   - 6 DoF (`joint1`..`joint6`), planning group `manipulator`, tip link
     `gripper_base` (there is no `link_tcp` frame: either add a fixed TCP frame at
     the fingertips with `RobotModel.with_fixed_frame` or put the fingertip
     offset into the grasp z), gripper joint `arm/gripper`, hardware range
     0..0.08 m (`GripperTask` normalises it to 0..1).
   - Wrist roll is `joint6`; pass `wrist_joint="joint6"` and `hand_links` /
     `elbow_links` that exist in the Piper URDF (`link6`, `gripper_base` for the
     hand; `link3`, `link4` for the elbow).
   - Reach is about 0.62 m, less than the xArm7: survey lower (0.30 to 0.35 m)
     and keep the container within 0.45 m of the base.
   - The adapter drives the arm to its zero pose on connect and on disconnect,
     and trajectories need `joint_names` set. Known CAN gotchas are in the
     `scene.py` docstring (`PIPER_JUDGE_CAN=0`, slcan bring-up at 1 Mbit).
2. **Check for a perception stack.** The xArm blueprint composes
   `ManipulationModule` + `ObjectSceneRegistrationModule` + `PointCloudSelfFilter`
   + `RayTracingVoxelMap` + `RealSenseCamera`. The Piper blueprints only have an RGB
   webcam. The skill needs a depth camera whose cloud reaches the planning frame:
   either a wrist RealSense (then do step 1) or a fixed depth camera with a
   measured extrinsic (then the rim fit works from one viewpoint and `survey` is
   a no-op: set `survey_height` to the current height and `survey_offset_xy` to
   zero).
3. **Decide the safety floor and boxes with the operator present.** On the xArm
   the operator hand-guided the fingertips to a safe height and that TCP z became
   `min_z`. Derive the boxes from the table: hand box = table extent plus a 10 cm
   margin behind the base, elbow box 20 cm larger laterally and tall enough for
   the survey posture (check with forward kinematics at the survey pose before
   the first run, otherwise every plan is rejected for the wrong reason).
4. **Controller limits.** The xArm SDK has `set_reduced_tcp_boundary`; piper_sdk
   has no TCP box. Compensate with lower speed scales (`plan_speed_scale` and
   `cartesian_speed_scale` 0.2), the module's guards, the Piper's own collision
   detection if enabled, and a hand on the e-stop for every unattended loop.

## 1. Calibrate perception (1 to 2 h with the tooling, more without)

- **Wrist camera hand-eye**: `dimos/manipulation/calibration/` has the ChArUco
  board, the AX=XB solver over all five OpenCV methods and an interactive
  blueprint (`xarm7-hand-eye-calibration`); the xArm run also used an autonomous
  driver that moved the arm through 30 poses per run over the SDK. Lessons:
  - measure the printed square (ours was 33.0 mm, not the nominal 34; the arm's
    own metric scale scan and the depth cross-check both caught it);
  - greedy nearest-neighbour pose ordering gives useless rotation diversity
    (0.2); order poses farthest-point in rotation space (0.47);
  - solve against the **URDF** forward kinematics of the frame the stack
    publishes (the SDK FK differed by 3 mm / 0.6 deg on the xArm);
  - expect 2 to 3 mm board spread; the offset along the optical axis is the weak
    direction. The result goes into the blueprint's static transform
    (`XARM_WRIST_CAMERA_TRANSFORM` on the xArm).
- **Scene camera**: factory intrinsics (ZED: `calib.stereolabs.com/?SN=<serial>`,
  the serial is on the HID interface, not the UVC one) plus one board view from
  the camera and the board's world pose from the hand-eye solve.

## 2. Compose the blueprint (30 min)

Copy `xarm_grasp_bin` in `dimos/robot/manipulators/xarm/blueprints/grasp.py` into a
Piper module: the manipulation stack, the camera, `PointCloudSelfFilter` with
`base_exclusion_radius` set to the base plate radius plus 5 cm, the mapper,
`ObjectSceneRegistrationModule`, `RimGraspModule(min_z=...)` and
`ContainerPickModule(model=..., hand_links=..., elbow_links=..., workspace_box=...,
elbow_box=..., wrist_joint="joint6", min_z=..., survey_height=...,
container_long_min=..., container_short_range=...)`. Register it with
`pytest dimos/robot/test_all_blueprints_generation.py`.

## 3. Bring-up checks, in this order (30 min)

Each of these failed once on the xArm and cost a loop iteration:

1. `app.ManipulationModule.plan_to_joints(<current joints + 1 mrad>)` succeeds.
   "Start configuration is in collision" means map cells inside the robot:
   raise `base_exclusion_radius` or check the static transforms.
2. `app.ContainerPickModule.survey()` succeeds and
   `app.PickAndPlaceModule.scan_objects(["bin"])` returns the object with a cloud
   of a few thousand points; `app.RimGraspModule.describe_rim(cloud)` gives the
   real rim size. If the rectangle is short or its centre is off the centroid,
   move the survey pose directly above the container.
3. Close the gripper on air and on the wall by hand once; set
   `grasp_verification.empty_epsilon` between the two readings.
4. One `pick_up_container()` with the operator watching. Then one
   `set_down_container()`. Confirm the container does not ride up after release
   (if it does, the along-wall slide direction is wrong: check the TCP yaw
   convention and the sign of the wrist joint, which on a down-pointing tool is
   opposite to the world yaw).

## 4. The research loop (1 h for 10 trials)

```bash
python -m dimos.manipulation.grasping.demo_container_pick_research out/ \
    --trials 10 --scene-camera /dev/videoN --hsv <lo_h,lo_s,lo_v,hi_h,hi_s,hi_v>
```

The schedule alternates wall selection (`lever`, `highest`, `lowest`, short
walls) and rotates the held container by 45 to 90 deg before each set-down, so
ten trials cover the yaw range. Judge success by the gripper readback **and** an
independent camera (the xArm run used a fixed ZED: the colour blob's top and
bottom must both rise by more than 20 px at ~2 mm/px). Read `trials.jsonl` after
every run; the failure modes seen so far, with their fixes:

| symptom in the log | cause | fix |
|---|---|---|
| `nothing in the jaws` on a wall candidate | partial segmentation, rim rectangle shrunk / off-centre | survey above the container; `container_long_min`, `rim_center_tolerance` |
| `REJECTED plan: path length 3+ rad` | 180 deg wrist flip or approach planned from far away | yaw modulo pi (built in); translate with Cartesian moves first (built in) |
| `REJECTED plan: link4 leaves its box` while standing still | elbow box too tight for the survey posture | compute FK at the survey pose; widen `elbow_box` or lower `survey_height` |
| `TOPP-RA ... initialGrid` | zero-length plan (already at the target) | treated as "already there" (built in) |
| container rides up after release | finger hooked under the lip | leave along the wall; check the slide axis sign |
| `Start configuration is in collision` | table cells under the base plate | `base_exclusion_radius` |
| planner detour / swing | obstacle cells from a carried object | mapping paused while holding (built in); guards reject the path |

## 5. Hand back

Report the trial table (yaw, wall, gripper readback, rise, outcome), the final
parameters, the calibration numbers, and which guards fired. Keep the per-trial
frames: they are what makes the next port faster.

Time budget that held on the xArm: calibration 2 h (first time), skill
bring-up 1 h, 16 trials 1 h. Most of the lost time was spent on motions that
were not checked before execution. Do the checks first.
