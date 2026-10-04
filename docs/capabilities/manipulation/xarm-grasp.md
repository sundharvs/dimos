# xArm Grasping

Two blueprints, differing only in which grasp provider they compose. Both carry
the control coordinator, the wrist camera, scene registration and
pick-and-place; both run on the real arm by default and switch to the MuJoCo
room scene with `--simulation`:

| Blueprint | Grasps |
|---|---|
| `xarm-grasp` | one top-down heuristic grasp, score 1.0 |
| `xarm-grasp-graspgenx` | up to 100 ranked learned grasps |

```bash
dimos run xarm-grasp-graspgenx --xarm7-ip 192.168.1.x     # hardware
dimos run xarm-grasp-graspgenx --simulation mujoco        # the room scene
```

Miss `--xarm7-ip` on hardware and the arm has no address to reach; leave
`--simulation` set and everything reverts to MuJoCo regardless of the IP.
`xarm-grasp-agent` and `xarm-grasp-graspgenx-agent` add an MCP agent over the
top; drive those with `dimos agent-send "..."`.

GraspGenX requires Linux x86_64, a CUDA 12.8-compatible GPU, and `uv >=0.9.25`.
The first launch prepares its isolated Python 3.12 environment and downloads the
checkpoints. Runtime sources come from the development checkout or the shared
repository clone used by installed dimOS.

To show the MuJoCo window with Rerun disabled:

```bash
MUJOCO_GL=glfw dimos --viewer none run xarm-grasp-graspgenx \
  --simulation mujoco --headless false
```

In another terminal, use `dimos shell` and follow [Driving it](#driving-it) to scan
objects and request grasps. See the [isolated-runtime development guide](/dimos/experimental/isolated_python/README.md#runtime-development)
for test and type-check commands.

What differs between the arm and the sim is decided at import time: the hardware
adapter, the base pose, the camera (RealSense plus its mount edge, versus the
MuJoCo wrist camera), the detector backends, and the home pose.

The manipulation viewer is on viser at `http://127.0.0.1:8095`. To watch the
MuJoCo scene itself, add `--headless false` with `MUJOCO_GL=glfw`. On a host
where `/dev/dri` must be hidden from Mesa, run inside the team's existing
`/dev/dri`-masked mount namespace:

```bash
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=true MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
  dimos --viewer none run xarm-grasp --simulation mujoco
```

## Calibrating the wrist camera

On hardware every detection reaches the world frame through
`XARM_WRIST_CAMERA_TRANSFORM`, the `link7 -> camera_link` mount edge in
`grasp.py`. Re-measure it whenever the camera mount moves. A rotation error in
the mount becomes a position error proportional to range, about 4.4 mm at 0.5 m
per half degree, so scanning from a raised pose needs a better calibration than
scanning from close up.

1. Print a ChArUco board and measure the printed square size with a rule:

   ```bash
   python -m dimos.manipulation.calibration.charuco --out charuco.png
   ```

2. Fix the board flat on the table, then run the calibration blueprint with
   the **measured** square size:

   ```bash
   dimos run xarm7-hand-eye-calibration --xarm7-ip 192.168.1.x --square-mm 34.0
   ```

3. Jog the arm in the Keyboard Teleop window. In the Hand-eye calibration
   window press SPACE to capture, U to undo and C to compute. Take 15 to 20
   poses, turning the wrist about a different axis at each one while keeping
   the board in view, and vary the distance around the range you will scan
   from. Translating the arm constrains nothing.

The module never commands the arm. Each capture requires the arm to have been
still and the board to reproject under 1 px. Rotation diversity is shown live;
below 0.15 the solve is refused, and above 0.4 is well spread.
3. Jog the arm in the Keyboard Teleop window until the board is in view, about
   as far away as you will scan from, then press A in the Hand-eye calibration
   window. The module drives the arm itself and solves at the end; X aborts the
   current motion.

   Or collect by hand: press SPACE to capture, U to undo and C to compute. Take
   15 to 20 poses, turning the wrist about a different axis at each one while
   keeping the board in view, and vary the distance. Translating the arm
   constrains nothing.

The automatic run first turns the wrist 8 degrees about link7's x and y axes and
15 degrees about z, both ways, which needs no knowledge of the mount and is
enough for a coarse solve. It then aims the camera at the board centre from 16
views on a cone about the board's normal: one ring tilted 25 degrees, one 12.5,
with the range varied 0.85 to 1.15 times the start range and up to 30 degrees of
roll. It returns to the start pose and solves. Moves are planned against the
robot model only, so stay at the arm: a view whose tool point would come within
8 cm of the board's plane, that is unreachable, or whose plan winds a joint
more than 90 degrees is skipped. The `--auto-*` flags tune all of this.

Each capture requires the arm to have been still and the board to reproject
under 1 px. Rotation diversity is shown live; below 0.15 the solve is refused,
and above 0.4 is well spread.

Computing runs all five OpenCV hand-eye solvers and keeps the one under which
the board's recovered base-frame pose is most consistent across captures. That
spread, in mm and degrees, is the number to judge the calibration by. The
solvers disagreeing by more than 5 mm means the data is thin. The result,
the per-method table and every capture go to
`~/.local/state/dimos/calibration/hand_eye.json`, and the report prints a
`~/.local/state/dimos/calibration/xarm7_wrist_realsense.json`, and the report prints a
`Transform(...)` to paste over `XARM_WRIST_CAMERA_TRANSFORM`. To re-solve the
saved captures offline:

```bash
python -m dimos.manipulation.calibration.hand_eye_module
python -m dimos.manipulation.calibration.hand_eye_module \
  ~/.local/state/dimos/calibration/xarm7_wrist_realsense_samples.json
```

### A fixed scene camera

`xarm7-side-camera-calibration` runs the same tool eye-to-hand for the fixed
side ZED 2i: mount the board rigidly on the gripper instead of the table, and
the result is `world -> camera_link` for that camera. The ZED is read as a
plain V4L2 webcam at 2K, so no ZED SDK is needed. Its raw left-camera
intrinsics come from the factory file the ZED SDK downloads for that serial,
copied to `~/.local/state/dimos/calibration/zed/SN33805648.conf`
(`--intrinsics-file` points elsewhere). A missing file, or one without the
frame's resolution, refuses captures rather than guessing intrinsics.

```bash
dimos run xarm7-side-camera-calibration --xarm7-ip 192.168.1.x --square-mm 34.0
```

## Voxel map obstacles

The wrist camera feeds a live voxel map that the planner treats as one octree
obstacle, so trajectories avoid whatever has actually been seen rather than only
the registered objects:

```
camera pointcloud
  -> PointCloudSelfFilter        drops the arm's own returns, emits a clear mask
  -> RayTracingVoxelMap          accumulates occupied cells in the world frame
  -> ManipulationModule.voxel_map   rebuilt as the "mapping/voxel-map" obstacle
```

`XARM_GRASP_VOXEL_SIZE` is the single resolution all three stages share; they
must agree or the clear mask names cells the map does not hold and the octree
does not line up with what was mapped. The blueprint also enables the camera's
`pointcloud` output, which is off by default on both the RealSense and the
MuJoCo camera, and publishes TF for every one of the arm's collision links. The
self filter drops a whole cloud if any link transform is missing at capture time.

Because the target object is itself mapped geometry, a collision-checked plan
into it can only ever be rejected. The pregrasp-to-grasp leg and the retreat are
therefore straight-line `move_linear` servos with collision checking off; only
the approach to the pregrasp pose is a checked plan.

## Seeing the proposals

The viser scene draws the ranked proposals as pose glyphs: an approach axis with
the closing axis across it, coloured best-green through worst-orange so the
ordering reads at a glance, with the top three drawn thicker and labelled with
their score. Only the leading twenty are drawn, because a hundred glyphs bury
the ranking they exist to show. `manipulation.grasp-proposals` in the Scene panel
toggles them.

The markers are pose indicators, not a gripper: what they promise is where a
grasp points and in what order the generator ranked it. To see what the arm will
actually do with one, watch the plan preview.

## The scene

The scene is an enclosed 2.6 m by 3.0 m room. The xArm is bolted to the world
origin. Unlike `data/xarm7`, this scene has no 12 cm pedestal, so the planning
model overrides `base_pose` to match. The 38 cm by 60 cm desk is in front of the
arm, with its work surface at `z=0.13 m`. Six scaled household targets sit on it:

| Object | Body position `(x, y, z)` m | Maximum grasp width | Approximate size |
|---|---:|---:|---:|
| Dark blue bottle | `(0.58, 0.19, 0.13)` | 6.0 cm | 6.0 × 6.0 × 17.6 cm |
| Gray can | `(0.545, -0.02, 0.13)` | 6.6 cm | 6.6 × 6.6 × 12.2 cm |
| Red cup | `(0.56, -0.22, 0.13)` | 6.8 cm | 6.8 × 6.8 × 6.1 cm |
| Green tape roll | `(0.38, 0.24, 0.13)` | 6.2 cm | 6.2 × 6.2 × 2.3 cm |
| Blue marker | `(0.35, 0.03, 0.13)` | 3.2 cm | 14.0 × 3.2 × 3.2 cm |
| Brown box | `(0.40, -0.19, 0.13)` | 6.0 cm | 8.4 × 6.0 × 4.5 cm |

The canonical positions and geometry live in `data/xarm_grasp_sim/scene.xml`.
The visual meshes were cooked from the `dimos_office` scene package and scaled
for the xArm gripper; primitive geometry is used only for contact. Every target
has a grasp axis below 7 cm, and the gray can is the designated pick smoke
target. The blueprint starts the arm at an elevated, collision-free top-down
scan pose so all six visual meshes fit in the wrist-camera frame.

OWL-ViT labels these synthetic renders unreliably. A scan routinely returns the
right six positions under swapped names, so match objects by position, not label.

## Driving it

In a second terminal, connect to the running blueprint:

```bash
dimos shell
```

Then run this complete scan and obstacle-inspection sequence:

```python skip
from dimos.robot.manipulators.xarm.blueprints.grasp import XARM_GRASP_PROMPTS

app.ManipulationSkills.go_init()
scan = app.PickAndPlaceModule.scan_objects(XARM_GRASP_PROMPTS)
print(scan)

print(app.ObjectSceneRegistrationModule.get_detected_objects())
print(app.ManipulationModule.refresh_obstacles())
print(app.ManipulationModule.get_obstacles())
```

CPU OWL-ViT inference takes about 11 seconds per prompt/frame on the validation
host, so let a scan finish rather than issuing another concurrently.

To pick, hand `pick_object` an `object_id` from that scan. Choose the target by
where its point cloud actually is rather than by name:

```python skip
scene = app.ObjectSceneRegistrationModule

for obj in scan.metadata["objects"]:
    cloud = scene.get_object_pointcloud_by_object_id(obj["object_id"])
    print(obj, cloud.points_f32().mean(axis=0) if cloud else None)

# the bottle sits at roughly (0.58, 0.19); pick whichever id landed there
pick = app.PickAndPlaceModule.pick_object("<object_id>")
print(pick)

app.PickAndPlaceModule.place_at(0.45, -0.25, 0.25)
```

`pick_object` generates the grasps itself, so there is no separate grasp call.
It opens the gripper, plans to the pregrasp, servos in, closes, verifies, and
retreats; with a learned provider it walks the ranked candidates until one is
reachable, and the result metadata carries the winning rank, its score and the
candidate count. To inspect grasps without moving the arm, call `propose_grasps`
on the provider directly:

```python skip
cloud = scene.get_object_pointcloud_by_object_id("<object_id>")
candidates = app.GraspGenXModule.propose_grasps(cloud)   # HeuristicGraspModule in the base blueprint
print(len(candidates.candidates), [c.score for c in candidates.candidates[:5]])
```

The prompt set includes a `green ring` fallback because the tape loses
its category silhouette in the wrist camera's top-down view.

A failed grasp knocks free-body targets out of place, and `MujocoSimModule.reset()`
does not respawn them. Restart the blueprint between pick attempts that need a
pristine scene.

## Recording demonstrations

`xarm-grasp-keyboard-collect` is the keyboard stack plus the imitation-learning
collection pair from `dimos/imitation/`: an `EpisodeMonitorModule` that segments
episodes from key presses, and a recorder that captures the colour and depth
images, the coordinator joint state, the operator's twist and gripper commands
and the episode markers into one session database.

```bash
dimos run xarm-grasp-keyboard-collect --xarm7-ip 192.168.1.x
```

Recording is continuous for the run; the keys only mark episodes. In the
"Keyboard Teleop" window:

| Key | Action |
| --- | --- |
| `SPACE` | Start an episode; press again to save it |
| `BACKSPACE` | Discard the episode in progress |

The window shows `● RECORDING` with the saved and discarded counts while an
episode is open. Picks and `move_near` calls issued from `dimos shell` during an
episode are recorded like any other motion. Sessions land in
`recordings/session_xarm7_grasp_<timestamp>.db` (or
`~/.local/state/dimos/recordings/` for an installed dimOS).

Export a LeRobot v3 dataset for ACT with the bundled config, after pointing its
`source` at the session:

```bash
dimos dataprep build -s recordings/session_xarm7_grasp_<timestamp>.db \
  -c dimos/robot/manipulators/xarm/blueprints/dataprep_xarm_grasp.json
dimos dataprep inspect data/datasets/xarm7_grasp
```

The config uses the colour image at 15 Hz as the anchor, the 8-D joint state
(seven joints plus gripper) as `observation.state`, and the next joint state as
`action`. The twist and gripper commands are in the database too for a config
that trains on commanded actions instead. Train with LeRobot (`lerobot-train
--policy.type=act`) and run the checkpoint with `LeRobotPolicyModule`
(`dimos/imitation/policy/lerobot/README.md`); its trajectory execution needs the
coordinator's trajectory task, which this stack already has.
