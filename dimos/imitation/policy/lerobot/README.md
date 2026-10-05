# LeRobot Policy Module

`LeRobotPolicyModule` runs trained LeRobot policies in a managed Python-native
subprocess. Its LeRobot, Transformers, Torch, and NumPy versions live in the
`native/python/lerobot` project and do not change the main DimOS environment.

The host contract subscribes to:

- `color_image: Image`
- `coordinator_joint_state: JointState`
- `button_pressed: Buttons`
- `teleop_buttons: Buttons`

It submits complete, timestamped action chunks to the existing
`joint_trajectory` task through the control coordinator. The state and action
vectors use `joint_names` order, including any gripper joint. The policy output
is already postprocessed into each joint's native absolute coordinate; the
runtime does not reinterpret gripper values.

```python
from dimos.imitation.policy.lerobot.module import LeRobotPolicyModule

policy = LeRobotPolicyModule.blueprint(
    policy_path="outputs/pick/checkpoints/last/pretrained_model",
    task="pick up the object",
    joint_names=["arm/joint1", "arm/joint2", "arm/gripper"],
    fps=30.0,
    robot_type="my_robot",
    image_width=640,
    image_height=480,
)
```

The module exposes `preflight_rollout`, `start_rollout`, `stop_rollout`, and
`rollout_status` RPCs. Preflight loads the checkpoint and processors, validates
the control task and fresh live observations, and sends no trajectory.
`start_rollout` refuses to run until preflight passes and rechecks observations
before starting.
The runtime rejects missing or stale observations, missing joints, non-finite
values, incompatible checkpoint features, and malformed action chunks. Pressing the
configured Quest button (A by default) toggles a preflighted rollout. Pressing
either middle-finger grip stops rollout. Release both grips before explicitly
restarting; releasing a grip alone never resumes the policy.

The runtime calls LeRobot's `predict_action_chunk()` every `replan_steps`
steps (default 1; `None` means once per `n_action_steps`), postprocesses the entire chunk, clips every action dimension
to the checkpoint's recorded data range, and folds it into a temporal ensemble
(ACT, Algorithm 2): each step's target is the `exp(-temporal_ensemble_coeff * i)`
weighted mean of every chunk that predicted it, `i = 0` for the oldest (default
coefficient 0.01, LeRobot's; `None` executes the newest chunk only;
`ensemble_window` caps how many of the newest chunks take part). Each
submission carries the ensemble's next `n_action_steps` targets at the
configured `fps` and lands while the previous trajectory is still running, so
the coordinator continues from its commanded position instead of stopping at
chunk boundaries. Trajectory execution uses the coordinator's existing
start-position and velocity handling. Configure `fps` to match the action
frequency used by the training dataset. At 15 fps each step has 66 ms for
inference plus the `execute_trajectory` RPC; a step that slips by whole periods
is covered by the previous submission's remaining targets and logged.

`image_feature` names the checkpoint's single camera input (default
`observation.images.wrist`; a dataset built from a `color_image` stream by
`dimos dataprep` has `observation.images.image`).

Every postprocessed action is sent as an absolute target in the hardware
joint's native coordinate, with one exception: set `gripper_joint` to a joint
in `joint_names` whose *action* is a normalized opening (0 closed .. 1 open)
while its *state* stays native, the convention of the DimOS collection
recorders. That joint is left out of the trajectory and the newest chunk's
first opening (a command, never averaged) is published on `gripper_command`
for the coordinator's gripper task whenever it changes.
`xarm-grasp-keyboard-policy` (`dimos/robot/manipulators/xarm/blueprints/grasp.py`)
is the reference setup.

Every rollout is logged as JSON lines under `rollout_log_dir` (default
`~/.local/state/dimos/policy_rollouts`, `None` disables): a header, every live
coordinator joint state, and one `chunk` record per inference with the full
predicted action chunk, the ensembled targets actually sent (`sent`), the step
index, the state it was predicted from, inference time and the coordinator's
answer. `rollout_status()` reports the current file as
`rollout_log`. Plot it against what the arm did, per joint plus a commanded-vs-
measured speed panel, with:

```bash
python -m dimos.imitation.policy.lerobot.tool_plot_rollout            # newest log -> PNG next to it
python -m dimos.imitation.policy.lerobot.tool_plot_rollout LOG --show --t0 5 --t1 20
```

`label` is a free-text tag written to the header; the xArm grasp blueprint
writes its `XARM_GRASP_POLICY_MODE` there. Score a set of logs by smoothness
and progress, grouped by that label, with:

```bash
python -m dimos.imitation.policy.lerobot.tool_compare_rollouts --since 19:30
```

Run isolated runtime checks with:

```bash
cd native/python/lerobot
uv run --isolated --locked --group tests --with-editable ../../.. python -m pytest
uv run --isolated --locked --group tests --with-editable ../../.. python -m mypy
```
