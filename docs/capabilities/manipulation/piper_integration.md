# Piper Integration

## Optional SLCAN setup

Use this separate path only with a serial-CAN adapter, such as `/dev/ttyACM0`;

```bash
sudo slcand -o -c -s8 /dev/ttyACM0 can0
sudo ip link set can0 up
```

This is a separate prerequisite for serial-CAN adapters. It is not needed when
the Piper adapter already exposes a native SocketCAN interface.

## Bring up a native Piper CAN interface

Piper uses SocketCAN at 1,000,000 bit/s. For the default vendor setup, use
the dimOS CLI to configure an existing CAN interface and bring it up:

```bash
dimos hardware can setup can0
```

For a non-default bitrate, pass `--bitrate` explicitly:

```bash
dimos hardware can setup can0 --bitrate 500000
```

The command prints each privileged operation before requesting sudo. Verify the
interface before starting a blueprint:

```bash
dimos hardware can status can0
```

## Run a Piper blueprint

Use the coordinator for the basic manipulation composition:

```bash
dimos --can-port can0 run coordinator-piper
```

For keyboard Cartesian teleoperation, use:

```bash
dimos --can-port can0 run keyboard-teleop-piper
```

The WebXR teleoperation composition is available as:

```bash
dimos --can-port can0 run teleop-webxr-piper
```

Note that ommitting the `--can-port` argument will fallback the control coordinator to use fake hardware adapter. This is good for testing.

## Serial CAN adapters (slcan) on a recent kernel

A CANable / CANable2 running the `slcan` serial firmware shows up as
`/dev/ttyACM0` and not as a CAN interface. Kernel 6.1 and newer ship a
SocketCAN-aware `slcan` driver, so attach the device and then configure it
like any other interface:

```bash
sudo modprobe slcan
sudo slcan_attach /dev/ttyACM0          # creates can0
dimos hardware can setup can0           # bitrate 1000000, txqueuelen, up
dimos hardware can status can0          # expect "bitrate 1000000"
candump can0                            # feedback frames once the arm is powered
```

If `ip -details link show can0` does not print a bitrate (older slcan
firmware set up with `slcand -s8`), `piper_sdk`'s start-up check rejects the
port. Skip that check with `PIPER_JUDGE_CAN=0` for the scene blueprints
below, or pass `adapter_kwargs={"judge_can": False}` to `make_piper_hardware`
in your own blueprint.

`slcan_attach` exits as soon as it has attached. If `can0` is gone when it
returns (seen on kernel 6.17 with a CANable2), use the `slcand` form from the
first section instead: the daemon stays resident and holds the interface, and
`PIPER_JUDGE_CAN=0` covers the bitrate it does not report.

Serial CAN is throughput-limited. If `candump` shows gaps or joint feedback
goes stale while the coordinator runs, flash the adapter with candleLight
firmware so it binds to `gs_usb` as a native SocketCAN device.

## Piper with an RGB scene camera

`piper-scene` and `piper-scene-coordinator` add a V4L2 camera to the
coordinator. Any UVC device works, including an Intel RealSense used for its
colour stream only (no librealsense or `pyrealsense2` needed):

```bash
PIPER_SCENE_CAMERA=/dev/video6 dimos --can-port can0 run piper-scene
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `PIPER_SCENE_CAMERA` | `/dev/video6` | V4L2 node or index of the RGB stream |
| `PIPER_SCENE_CAMERA_WIDTH` / `_HEIGHT` | `1280` / `720` | capture size |
| `PIPER_SCENE_CAMERA_FPS` | `15` | software frame-rate cap |
| `PIPER_JUDGE_CAN` | `1` | set `0` to skip the piper_sdk CAN self-check |

For a RealSense, the by-id `video-index0` link is the depth node; use the
`/dev/videoN` or by-path node that OpenCV opens as YUYV.

`piper-scene` includes the Drake planner (`plan_to_joints`, `plan_to_poses`,
`move_linear`, `execute` RPCs plus a viser model view). `piper-scene-coordinator`
is the lighter variant with only joint trajectories and the gripper task.
Frames appear in Rerun under `world/color_image`.

Sending commands once running:

```bash
dimos shell
>>> app.ControlCoordinator.get_joint_positions()
>>> app.ControlCoordinator.task_invoke("arm_gripper", "set_normalized", {"values": [1.0]})
>>> app.ControlCoordinator.set_estop(True)          # software stop

uv run python -m dimos.manipulation.control.coordinator_client   # joint moves in degrees
```

The Piper adapter moves every joint to the zero pose when it connects and
again before it disables the arm on shutdown. Clear the workspace first.

### The arm connects but does not move

Check the control mode byte of the arm's status frame:

```bash
candump can0,2A1:7FF | head -1     # first data byte: 00 standby, 01 CAN control, 02 teaching
```

In teaching mode (`02`) the arm reports its motors as enabled yet ignores
every joint command, and the adapter logs an error at connect. Only the mode
button on the arm's base leaves that mode: put the arm in standby (solid
green LED), then restart the blueprint. The adapter's connect sequence
(reset, enable, CAN control mode) then takes over. Note that on enable the
arm immediately executes the last joint target it latched, so leave it at
rest before restarting.

A joint whose driver has latched a fault is the other cause. After a collision
or an overheat that joint is disabled, and the arm then ignores every joint
command while trajectories still report completed. Each joint's low-speed
feedback frame carries a driver status byte:

```bash
candump can0,260:7F8      # 0x261-0x266, sixth data byte: 40 is enabled and healthy
```

Anything else on a joint while the others read `40` is a fault; `32`, for
one, is motor overheat, collision and driver error with the enable bit clear.
(All six reading `30` before any blueprint has connected is only the state
after power-up.) Stop the blueprint, clear the joint's error and start again.

## Grasping with the wrist camera

`piper-grasp` is the `xarm-grasp` stack on a Piper with a wrist RealSense D405:
planner, `ManipulationSkills`, `PickAndPlaceModule`, scene registration with
moondream + EdgeTAM, and the heuristic grasp provider.

```bash
PIPER_JUDGE_CAN=0 PIPER_JOINT_OFFSETS_DEG=0,0,0,0,4.42,0 dimos --can-port can0 run piper-grasp
```

What is particular to this arm:

- Poses are planned to a tool frame between the jaws (`gripper_tcp`, 11.8 cm
  along `gripper_base`, 2 cm in from the fingertips), not to `gripper_base`, so
  a height passed to `move_to_pose` is where the grasp point goes.
- `go_home` moves to the scan pose: the wrist camera looking down at the table
  from about 22 cm.
- The wrist pitch range bounds where the gripper can point straight down, and
  the band narrows with height, so the pre-grasp stops 6 cm above a grasp.
- The wrist camera is a D405, whose depth unit is 0.1 mm; scene registration is
  configured for it.
- Joint 6 spans +-120 deg, so top-down grasps take the half-turn-equivalent yaw.
- Commanded fully open, the jaws stop at about 67 mm of the 80 mm commanded.
- The camera mount, the table height and the joint offsets are measured values
  in `dimos/robot/manipulators/piper/blueprints/grasp.py` and
  `PIPER_JOINT_OFFSETS_DEG`. Re-measure them when the rig moves; the camera edge
  comes from `piper-hand-eye-calibration`.
