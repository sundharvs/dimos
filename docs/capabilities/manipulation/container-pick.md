# Container pick (rim grasp)

A parallel-jaw gripper cannot span a bin, and a centroid grasp on an open
container closes on air. The container pick skill grasps one **wall** by the rim
from above, lifts, and later lowers, opens and backs the open jaws out along the
wall before lifting away. It was learned on an xArm7 with a wrist RealSense on
2026-10-03: 15 clean lifts in 16 attempts at 13 different bin orientations, on
long, short, high and low (scoop-front) walls.

Two modules carry it:

- `RimGraspModule` (`dimos/manipulation/grasping/rim_grasp.py`), a `GraspGenSpec`
  provider. From the object's point cloud it takes the top band of points, fits a
  rotated rectangle to the rim, and proposes one top-down grasp per wall: TCP
  `insertion_depth` below that wall's own top, jaw closing axis normal to the
  wall. Candidates are ranked by `wall_select`: `lever` (closest to the
  footprint centroid, the default), `highest`, `lowest` (the opening of a
  scoop-front bin) or `normal` (nearest a preferred outward direction).
- `ContainerPickModule` (`dimos/manipulation/container_pick_module.py`), the skills
  `survey`, `pick_up_container`, `rotate_held_container` and `set_down_container`,
  with the motion guards described below. It needs a `ManipulationSpec`, an
  `ObjectSceneRegistrationSpec`, a `GraspGenSpec` and, optionally, a self filter
  that can pause the obstacle map while an object is held.

## Run it on the xArm7

```bash
dimos --viewer none run xarm-grasp-bin --xarm7-ip 192.168.1.197 --daemon
dimos shell
```

```python
app.ContainerPickModule.pick_up_container()          # scan, grasp a wall, lift 25 cm
app.ContainerPickModule.rotate_held_container(45)    # optional, guarded wrist move
app.ContainerPickModule.set_down_container()         # lower, open, back out, lift
app.ContainerPickModule.status()
app.RimGraspModule.set_params(wall_select="lowest", insertion_depth=0.03)
```

Before the first unattended run on the xArm, put a TCP box in the controller.
It persists across power cycles and is the first line of defence:

```python
from xarm.wrapper import XArmAPI
arm = XArmAPI("192.168.1.197")
arm.set_reduced_tcp_boundary([350, -350, 50, -750, 900, 158])  # x+ x- y+ y- z+ z- in mm
arm.set_reduced_mode(True)
```

## What the guards do, and why

On 2026-10-03 an unattended loop asked RoboPlan's RRT for a return path from a
stretched pose while the voxel map held cells from the carried bin; the planner
returned a 2.9 s swing up and over the base and the operator hit the e-stop.
Since then every planned motion goes through `PathGuard` before `execute`:

- forward kinematics of **every waypoint** keeps `hand_links` inside
  `workspace_box` and `elbow_links` inside `elbow_box`;
- `base_joint` turns at most `base_joint_max_excursion` (1 rad) along one path;
- the joint-space path length is at most `path_max_length` (3 rad);
- a rejected plan is cleared and never executed.

Translations (approach, descent, lift, carry, return) are straight-line
Cartesian moves whose targets are box-, reach- and `min_z`-checked first; the
planner is only asked to rotate the wrist in place, and the grasp yaw is taken
modulo a half turn so the wrist never flips. Planned and Cartesian speeds are
scaled to 30 %. The self filter excludes the base-plate footprint from the map
(`base_exclusion_radius`), otherwise table cells under the base put `link_base`
in collision and the planner refuses every start configuration, and it is
paused while an object is held so the carried object is not mapped as an
obstacle along the carry.

## What was learned about the grasp

- Insertion 3 cm below the wall top, no across offset, works on every wall of a
  28 x 10 x 9 cm plastic shelf bin. Long walls have the smallest lever arm and the
  most reliable rim fit; short walls (10 cm) are the only place a miss happened.
- A thin wall reads 0.02 to 0.045 of normalised gripper travel on the xArm; air
  reads about 0.0. `GraspVerificationConfig.empty_epsilon` must sit between them
  (0.012 here), or a good grasp is reported as empty and reopened.
- The jaws often slide up the wall on the first lift until the rim lip stops
  them; the bin then hangs from the lip and still comes up level. Lowering back to
  the grasp height on set-down lets the bin touch first and the jaws slide down.
- Release must leave **along** the wall past the container's end. An open finger
  left inside a bin hooks the lip and the bin rides up with the arm, whichever
  way the gripper is shifted across the wall.
- From a survey pose beside the container the segmentation is often partial and
  the rim rectangle shrinks and drifts; grasps from it close on air. Survey
  straight above the container and validate the rectangle
  (`container_long_min`, `container_short_range`, `rim_center_tolerance`) before
  trusting it.
- Vary the container's orientation between trials. One orientation hides wall
  classification and release bugs.

## The research loop

`dimos/manipulation/grasping/demo_container_pick_research.py` runs the trial
schedule (wall selection x rotation after each lift) against a live stack, judges
each lift by the gripper readback and, when a fixed scene camera is given, by the
object's colour blob rising in it, and writes `trials.jsonl` plus a frame per
step. Porting notes for another arm: [Porting the container pick to the AgileX Piper](/docs/capabilities/manipulation/porting-container-pick-piper.md).
