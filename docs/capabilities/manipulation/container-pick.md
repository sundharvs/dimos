# Container pick and place (rim grasp)

A parallel-jaw gripper cannot span a bin, and a centroid grasp on an open
container closes on air. The container skills grasp one **wall** by the rim from
above, lift, turn the hanging container so its opening faces where it should,
carry it over the target, lower it, open and back the open jaws out over the
container's low end before lifting away. Learned on an xArm7 with a wrist
RealSense on 2026-10-03: 15 clean lifts in 16 attempts at 13 bin orientations,
then 9 of 9 placements of the bin into masking-tape slots (left, middle, right)
with the opening toward the arm, 8 on the first try and one after a re-pick,
with the bin scrambled by up to 120 degrees before each placement.

Three modules carry it:

- `RimGraspModule` (`dimos/manipulation/grasping/rim_grasp.py`), a `GraspGenSpec`
  provider. From the object's point cloud it takes the top band of points, fits a
  rotated rectangle to the rim, and proposes one top-down grasp per wall: TCP
  `insertion_depth` below that wall's own top, jaw closing axis normal to the
  wall. Candidates are ranked by `wall_select`: `lever` (closest to the
  footprint centroid, the default), `highest`, `lowest` (the opening of a
  scoop-front bin) or `normal` (nearest a preferred outward direction).
- `WristTabletopModule` (`dimos/manipulation/wrist_tabletop_module.py`), a
  `WristTabletopSpec` provider on the wrist camera's aligned colour + depth
  frames: `scan_object_cloud` segments the container by colour and back-projects
  it into the planning frame (no memory, unlike the scene registry, which
  accumulates the cloud of anything re-seen within 5 cm of a stored object and so
  corrupts repeated placements), and `add_tape_view` / `fit_slots` map masking
  tape on the table into a named grid of slots.
- `ContainerPickModule` (`dimos/manipulation/container_pick_module.py`), the skills
  `survey`, `pick_up_container`, `rotate_held_container`, `place_container`,
  `set_down_container`, `check_container_pose`, `map_slots` and
  `place_container_in_slot`, with the motion guards described below. It needs a
  `ManipulationSpec`, an `ObjectSceneRegistrationSpec`, a `GraspGenSpec` and,
  optionally, a `WristTabletopSpec` (preferred for scans when present) and a self
  filter that can pause the obstacle map while an object is held.

## Run it on the xArm7

```bash
dimos --viewer none run xarm-grasp-bin --xarm7-ip 192.168.1.197 --daemon
dimos shell
```

```python
app.ContainerPickModule.map_slots()                        # wrist camera over the tape, fit the grid
app.ContainerPickModule.place_container_in_slot("left")    # pick, turn, carry, set down, verify, correct
app.ContainerPickModule.pick_up_container(turn_after_degrees=90)  # scan, grasp a wall, lift 25 cm
app.ContainerPickModule.place_container(-0.22, -0.55, opening_yaw_degrees=90)  # opening toward +Y
app.ContainerPickModule.check_container_pose()             # centre, heading, opening direction
app.ContainerPickModule.set_down_container()               # back where it was picked
app.ContainerPickModule.status()
app.RimGraspModule.set_params(wall_select="lowest", insertion_depth=0.03)
app.WristTabletopModule.get_slots()
```

Slot names run from -X to +X (`left`, `middle`, `right` as seen from the table's
front, where the scene camera stands); the opening is placed toward the slot end
nearer the base. The tape's HSV range, the container's colour and the camera
viewpoints for `map_slots` are blueprint parameters (`xarm-grasp-bin`).

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
- the joint-space path length is at most `path_max_length` (3 rad), except for a
  pure wrist turn, which may be a half turn or more;
- a rejected plan is cleared and never executed, and a planner exception clears
  the pending plan so the manipulation module does not stay in PLANNING and
  refuse every later command (seen when a wrist goal exceeded the joint limit).

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

## What was learned about placing

- **The opening is the lower end wall, measured in the middle of the end.** A
  scoop-front bin's front is cut down 4 cm, but the rim fitter's per-wall top
  includes the corners, which belong to the full-height side walls, so it read
  the drop as 0 to 4 cm at random and once turned the bin the wrong way round.
  `describe_container` takes the 90th percentile height of the points within
  3 cm of each end and the middle 60 % of the width; a drop under
  `opening_min_drop` (1.2 cm) means "unknown", and the last known direction is
  kept rather than guessed.
- **The container lands closer to the jaws than the rim says.** Hanging from one
  wall it tilts, and on touch-down its centre ends 4.0 cm from the TCP instead of
  the 5.3 cm rim half width, consistently, on either side. `landing_offset` holds
  the learned value; with it, placements land within 1 cm.
- **Exit over the opening.** After the jaws open, the finger inside the container
  cannot pass a full-height end wall: backing out over one pushed the bin 9.5 cm
  along. Backing out over the low scoop end, 2 cm up and 5 cm past the end, never
  moved it. `set_down_container` falls back to the old along-the-wall exit only
  when the opening is unknown.
- **Choose the grasp yaw for the turn that follows.** The wrist joint has +-3.1
  rad on the xArm7, not the full turns a tool-pointing-down pose suggests. The
  grasp yaw equivalent (yaw or yaw + pi) is chosen so the wrist is inside its
  limits both at the grasp and after the planned turn (`wrist_feasible_yaw`).
- **Survey with the camera, not the TCP, above the target.** The wrist camera sits
  7 cm from the TCP; at the mirrored survey yaw that offset flips and half the
  container leaves the frame. `survey_camera_offset` lets the module put the
  optical centre above the target at whichever survey yaw is nearer.
- Verification from above (rim corners inside the tape's inner edges, opening
  toward the base) plus a re-pick with the measured error folded into the next
  target fixes the occasional 3 cm miss in one extra round.

## The research loops

`dimos/manipulation/grasping/demo_container_pick_research.py` runs the lift trial
schedule (wall selection x rotation after each lift) against a live stack, judges
each lift by the gripper readback and, when a fixed scene camera is given, by the
object's colour blob rising in it, and writes `trials.jsonl` plus a frame per
step. The placement research (scramble, pick, turn, place into a slot, verify from
above, correct) was driven by the scripts kept with its artifacts under
`~/.local/state/dimos/research/xarm7_bin_place_2026-10-03/` (README there). Porting notes for another arm: [Porting the container pick to the AgileX Piper](/docs/capabilities/manipulation/porting-container-pick-piper.md).

## On the AgileX Piper

`dimos run piper-grasp-bin --can-port can0` composes the same skills for the
6-DoF Piper (`dimos/robot/manipulators/piper/blueprints/grasp.py`). What differs
is configuration, each number with its measurement in the blueprint:

- **Survey from a joint pose** (`survey_joints`): the Piper's wrist pitch range
  lets the tool point straight down only below a TCP height of about 12.5 cm, at
  radii of 0.16 to 0.32 m, so there is no top-down survey pose.
- **One planned move to the pre-grasp** (`planned_approach`), trying both
  equivalent jaw yaws, instead of straight legs and a wrist turn in place.
- **Lift, then lean** (`lift_joint_offsets`): a 4 cm straight lift, then joint 2
  leans back 0.35 rad, which raises the hand to about 0.20 m. The lean is undone
  before a set-down or a turn.
- **Grip at 4 N*m** (`PIPER_BIN_GRIPPER_EFFORT`): at the adapter's 1 N*m the bin's
  2 mm wall pivots in the pads and slides out.
- **The hold is judged by the wrist camera** (`hold_check="camera"`): the jaw
  readback is 0.002 with the bin hanging in the jaws. After the lift the colour
  scan must show the bin risen with the hand; a bin hanging closer than the depth
  range must fill the colour frame instead.
- **Leave straight up** after a set-down (`release_exit="up"`).

Validation and what is untested: `PIPER_BIN_VALIDATION` in the blueprint. Known
limits: a bin lying along the arm's X axis around x 0.31 m had no IK solution for
either jaw yaw; and the pre-grasp must fall inside the top-down band above.
