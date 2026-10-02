# XR Headset Body-Tracking Teleop

Teleoperate an arm from an XR headset's full-body and hand tracking, received
through [xr-robot-teleop-server](https://pypi.org/project/xr-robot-teleop-server/).
The right wrist flies the TCP and the left hand signs commands.

```bash
uv sync --extra xr-server --inexact
dimos --simulation run teleop-xr-server-xarm7   # MuJoCo
dimos run teleop-xr-server-xarm7                # real xArm7
```

Point the headset app at `http://<robot-pc-ip>:8080/offer`. The headset and
robot PC must share a network with TCP 8080 open. Only one client may connect
per headset.

## How it maps to the robot

`XrServerTeleopModule` publishes the same ports as the WebXR `ArmTeleopModule`,
so it uses the same `TeleopIKTask` in the coordinator:

- The right-wrist pose is expressed in the operator's **body frame**, which is
  re-estimated every message from the shoulders and spine. It is then scaled by
  `position_scale` and yawed by `body_yaw_deg`.
- **Latching** sets the `right_grip` deadman. On the rising edge
  `TeleopIKTask` anchors the current wrist pose to the current TCP pose. After
  that the target is absolute: `p_tcp0 + Δp_hand` and `ΔR_hand · R_tcp0`, with
  the rotation delta applied about base axes. A dropout cannot integrate into
  drift.
- If no skeleton arrives for `stale_timeout_s` (0.5 s), the latch is released
  and the arm holds. Re-latch to continue.
- If the hand frame drops out while latched, orientation holds the last value
  that was sent. The gripper always holds when data is missing; it never opens.

## Left-hand signs

A sign fires once, after being held for 1 s. Fist and open hand are not
commands, so the gripper holds while you sign.

| Sign | Action |
|------|--------|
| index only | latch / re-latch |
| index + middle | start take (latches first) / save take (then releases and homes) |
| three fingers | **stop and home**: release the latch, open the gripper, move to `home_joints_deg` |
| pinky only | discard take, release and home |
| fist / open hand | close / open gripper (`gripper_hand="left"`) |

Take start/save and discard pulse `B` and `Y` on `teleop_buttons`, which is
`EpisodeMonitorModule`'s default button map. From `dimos shell` you can also
call `latch()` and `release()`.

Homing is a blocking planned move through `ManipulationModule` (plan to
`home_joints_deg`, then execute). The latch is refused until it finishes. With
`home_joints_deg` unset, or no `ManipulationModule` in the blueprint, these
signs only release. `home()` is also callable from `dimos shell`.

## Configuration

| Field | Default | Notes |
|-------|---------|-------|
| `server_port` | 8080 | WebRTC signalling (`/offer`) |
| `control_loop_hz` | 20 | Rate at which poses and buttons are published ; `teleop-xr-server-xarm7` sets 100 |
| `ema_tau_s` | 0.15 | Smoothing of the published pose; reset on every latch, 0 disables |
| `position_scale` | 1.0 | Bring-up value; the reference rig used 2.2 |
| `body_yaw_deg` | 0 | Operator facing relative to robot +x. `teleop-xr-server-xarm7` sets −90 (operator faces the robot's −Y) |
| `gestures` | true | Enables the left-hand sign commands |
| `home_joints_deg` | none | Home joint pose in degrees; none disables homing |
| `gripper_hand` | `left` | `right` = thumb–index pinch (> 5 cm opens) |

## Bring-up

1. **Receiver only:** `DIMOS_LOG_LEVEL=DEBUG dimos run demo-xr-server-receiver`.
   Check that the left-hand signs classify correctly in `dimos log -f`.
2. **Sim:** `dimos --simulation run teleop-xr-server-xarm7`. Latch, then check
   that moving your hand forward, left and up moves the TCP +x, +y and +z.
   Check that a yaw of your wrist yaws the TCP in the same direction.
3. **Hardware, gently:** keep `position_scale=1.0`, keep a hand on the e-stop,
   and make small moves.
