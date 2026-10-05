# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavior tests for the isolated LeRobot runtime."""

from collections.abc import Callable, Iterator
import json
from pathlib import Path
from threading import Event, Thread
import time
from typing import Any, Protocol

from dimos_lerobot import runtime as policy_runtime
from dimos_lerobot.runtime import LeRobotPolicyRuntime, _ActionEnsemble
from lerobot.configs.policies import PreTrainedConfig
import numpy as np
from numpy.typing import NDArray
import pytest
import pytest_mock
import torch
from torch import Tensor

from dimos.control.tasks.trajectory_task.trajectory_task import (
    TrajectoryCancellationResult,
    TrajectoryCancellationStatus,
    TrajectoryExecutionResult,
    TrajectoryExecutionStatus,
)
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.protocol.rpc.pubsubrpc import LCMRPC
from dimos.teleop.webxr.controller_types import Buttons
from dimos.utils.testing.waiting import wait_until

JOINTS = [f"test_arm/joint{i}" for i in range(1, 5)]


class FakeFeature:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class FakeUpstreamConfig:
    def __init__(self, joint_count: int, n_action_steps: int) -> None:
        self.type = "fake_policy"
        self.device: str | None = "cpu"
        self.use_amp = False
        self.chunk_size = 3
        self.n_action_steps: int | None = n_action_steps
        self.temporal_ensemble_coeff: float | None = None
        self.input_features = {
            "observation.images.wrist": FakeFeature((3, 4, 5)),
            "observation.state": FakeFeature((joint_count,)),
        }
        self.output_features = {"action": FakeFeature((joint_count,))}


class FakePipeline:
    def __init__(self, steps: list[object] | None = None) -> None:
        self.calls: list[object] = []
        self.reset_count = 0
        self.steps = steps or []

    def __call__(self, value: object) -> object:
        self.calls.append(value)
        return value

    def reset(self) -> None:
        self.reset_count += 1


class FakeActionStats:
    def __init__(self, lower: list[float], upper: list[float]) -> None:
        self._state = {
            "action.min": torch.tensor(lower),
            "action.max": torch.tensor(upper),
        }

    def state_dict(self) -> dict[str, Tensor]:
        return self._state


class FakePolicy:
    def __init__(
        self,
        action_chunk: NDArray[np.float32],
        n_action_steps: int = 2,
        *,
        action_lower: list[float] | None = None,
        action_upper: list[float] | None = None,
    ) -> None:
        self.action_chunk = torch.from_numpy(action_chunk).unsqueeze(0)
        self.called = Event()
        self.reset_count = 0
        self.batch: dict[str, object] | None = None
        self.upstream_config = FakeUpstreamConfig(len(JOINTS), n_action_steps)
        self.preprocessor = FakePipeline()
        self.postprocessor = FakePipeline(
            [
                FakeActionStats(
                    action_lower or [-100.0] * len(JOINTS),
                    action_upper or [100.0] * len(JOINTS),
                )
            ]
        )
        self.config_load_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(self, batch: dict[str, object]) -> Tensor:
        self.batch = dict(batch)
        self.called.set()
        return self.action_chunk


class RuntimeFactory(Protocol):
    def __call__(
        self,
        policy: FakePolicy,
        *,
        device: str | None = None,
        **config: Any,
    ) -> tuple[LeRobotPolicyRuntime, Any]: ...


@pytest.fixture
def make_runtime(mocker: pytest_mock.MockerFixture) -> Iterator[RuntimeFactory]:
    mocker.patch("dimos.core.module.get_loop", return_value=(mocker.MagicMock(), None))
    mocker.patch.object(LCMRPC, "__init__", return_value=None)
    mocker.patch.object(LCMRPC, "serve_module_rpc", return_value=None)
    mocker.patch.object(LCMRPC, "start", return_value=None)
    mocker.patch.object(LCMRPC, "stop", return_value=None)
    built: list[LeRobotPolicyRuntime] = []

    def _make(
        policy: FakePolicy,
        *,
        device: str | None = None,
        **config: Any,
    ) -> tuple[LeRobotPolicyRuntime, Any]:
        def load_config(_path: str) -> FakeUpstreamConfig:
            policy.config_load_count += 1
            return policy.upstream_config

        policy_class = mocker.MagicMock()
        policy_class.from_pretrained.return_value = policy
        mocker.patch.object(PreTrainedConfig, "from_pretrained", side_effect=load_config)
        mocker.patch.object(policy_runtime, "get_policy_class", return_value=policy_class)
        mocker.patch.object(
            policy_runtime,
            "make_pre_post_processors",
            return_value=(policy.preprocessor, policy.postprocessor),
        )

        def prepare_observation(
            observation: dict[str, NDArray[np.uint8] | NDArray[np.float32]],
            _device: torch.device,
            *,
            task: str,
            robot_type: str,
        ) -> dict[str, object]:
            return {
                **{
                    name: torch.from_numpy(value).unsqueeze(0)
                    for name, value in observation.items()
                },
                "task": task,
                "robot_type": robot_type,
            }

        mocker.patch.object(
            policy_runtime,
            "prepare_observation_for_inference",
            side_effect=prepare_observation,
        )
        mocker.patch.object(policy_runtime, "register_third_party_plugins")

        module = LeRobotPolicyRuntime(
            _isolated_python_runtime=True,
            policy_path="checkpoint/default",
            task="pick up the test object",
            device=device,
            joint_names=JOINTS,
            fps=config.pop("fps", 50.0),
            robot_type="test_arm",
            image_width=5,
            image_height=4,
            # Never write fake rollouts into the user's real log directory.
            **{"rollout_log_dir": None, **config},
        )
        control = mocker.MagicMock()
        control.execute_trajectory.return_value = TrajectoryExecutionResult(
            TrajectoryExecutionStatus.ACCEPTED
        )
        control.cancel_trajectory.return_value = TrajectoryCancellationResult(
            TrajectoryCancellationStatus.ALREADY_STOPPED
        )
        control.list_tasks.return_value = ["joint_trajectory"]
        mocker.patch.object(module, "_control", control, create=True)
        built.append(module)
        return module, control

    yield _make
    for module in built:
        module.stop()


def _action_chunk(steps: int = 3) -> NDArray[np.float32]:
    return np.arange(steps * len(JOINTS), dtype=np.float32).reshape(
        steps, len(JOINTS)
    ) / np.float32(10)


def _provide_observation(
    module: LeRobotPolicyRuntime,
    *,
    positions: list[float] | None = None,
    ts: float | None = None,
) -> tuple[NDArray[np.uint8], list[float], float]:
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 30
    values = positions or [float(i) / 10 for i in range(len(JOINTS))]
    timestamp = time.time() if ts is None else ts
    module._on_color_image(Image(data=rgb, format=ImageFormat.RGB, ts=timestamp))
    module._on_joint_state(JointState(ts=timestamp, name=JOINTS, position=values))
    return rgb, values, timestamp


def _preflight(module: LeRobotPolicyRuntime) -> None:
    status = module.preflight_rollout()
    assert status["policy_ready"] is True
    assert status["observations_ready"] is True
    assert status["last_error"] is None


def test_policy_predicts_and_executes_one_native_joint_chunk(make_runtime: RuntimeFactory) -> None:
    actions = _action_chunk()
    policy = FakePolicy(actions, n_action_steps=2)
    module, control = make_runtime(policy)
    rgb, positions, _ts = _provide_observation(module)

    _preflight(module)
    assert module.start_rollout()["active"] is True
    wait_until(lambda: control.execute_trajectory.call_count >= 1, timeout=1.0)
    module.stop_rollout()

    call = control.execute_trajectory.call_args_list[0]
    trajectory = call.args[0]
    assert call.kwargs == {}
    assert trajectory.joint_names == JOINTS
    assert [point.time_from_start for point in trajectory.points] == [0.0, 0.02, 0.04]
    np.testing.assert_allclose(trajectory.points[0].positions, positions)
    np.testing.assert_allclose(trajectory.points[1].positions, actions[0])
    np.testing.assert_allclose(trajectory.points[2].positions, actions[1])
    assert policy.batch is not None
    assert policy.batch["task"] == "pick up the test object"
    image = policy.batch["observation.images.wrist"]
    assert isinstance(image, Tensor)
    np.testing.assert_array_equal(image.squeeze(0).numpy(), rgb)
    assert policy.postprocessor.calls
    assert module.rollout_status()["chunks_accepted"] >= 1


def test_gripper_joint_goes_to_the_gripper_stream_not_the_trajectory(
    make_runtime: RuntimeFactory, mocker: pytest_mock.MockerFixture
) -> None:
    actions = _action_chunk()
    actions[:, -1] = [1.5, 0.0, 0.25]  # normalized opening; the first step is clipped to 1.0
    policy = FakePolicy(actions, n_action_steps=2)
    policy.upstream_config.input_features["observation.images.image"] = (
        policy.upstream_config.input_features.pop("observation.images.wrist")
    )
    module, control = make_runtime(
        policy, image_feature="observation.images.image", gripper_joint=JOINTS[-1]
    )
    gripper = mocker.MagicMock()
    mocker.patch.object(module, "gripper_command", gripper, create=True)
    _rgb, positions, _ts = _provide_observation(module)

    _preflight(module)
    assert module.start_rollout()["active"] is True
    wait_until(lambda: control.execute_trajectory.call_count >= 1, timeout=1.0)
    module.stop_rollout()

    trajectory = control.execute_trajectory.call_args_list[0].args[0]
    assert trajectory.joint_names == JOINTS[:-1]
    np.testing.assert_allclose(trajectory.points[0].positions, positions[:-1])
    np.testing.assert_allclose(trajectory.points[1].positions, actions[0, :-1])
    np.testing.assert_allclose(trajectory.points[2].positions, actions[1, :-1])
    assert gripper.publish.call_count >= 1
    assert gripper.publish.call_args_list[0].args[0].data == pytest.approx(1.0)
    assert policy.batch is not None
    assert "observation.images.image" in policy.batch
    assert "observation.images.wrist" not in policy.batch


def test_rollout_writes_a_jsonl_log_of_chunks_and_joint_states(
    make_runtime: RuntimeFactory, tmp_path: Path
) -> None:
    policy = FakePolicy(_action_chunk(), n_action_steps=2)
    module, control = make_runtime(policy, rollout_log_dir=str(tmp_path / "logs"))
    _provide_observation(module)

    _preflight(module)
    status = module.start_rollout()
    assert status["active"] is True
    assert status["rollout_log"] is not None
    log_path = Path(status["rollout_log"])
    wait_until(lambda: control.execute_trajectory.call_count >= 1, timeout=1.0)
    _provide_observation(module)
    module.stop_rollout()
    wait_until(lambda: module.rollout_status()["rollout_log"] is None, timeout=1.0)

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    kinds = [r["type"] for r in records]
    assert kinds[0] == "header"
    assert records[0]["joint_names"] == JOINTS
    assert records[0]["n_action_steps"] == 2
    assert "joint_state" in kinds
    chunk = next(r for r in records if r["type"] == "chunk")
    assert len(chunk["chunk"]) == 3 and len(chunk["chunk"][0]) == len(JOINTS)
    assert chunk["executed_steps"] == 1
    assert chunk["step"] == 0
    assert chunk["sent"] == chunk["chunk"][:2]
    assert chunk["result"] == "ACCEPTED"
    assert records[0]["replan_steps"] == 1
    assert records[0]["label"] == ""
    assert records[0]["temporal_ensemble_coeff"] == pytest.approx(0.01)
    assert kinds[-1] == "end"


def test_rollout_log_can_be_disabled(make_runtime: RuntimeFactory) -> None:
    policy = FakePolicy(_action_chunk(), n_action_steps=2)
    module, control = make_runtime(policy, rollout_log_dir=None)
    _provide_observation(module)
    _preflight(module)
    assert module.start_rollout()["rollout_log"] is None
    wait_until(lambda: control.execute_trajectory.call_count >= 1, timeout=1.0)
    module.stop_rollout()


def test_policy_actions_are_clipped_to_checkpoint_range(make_runtime: RuntimeFactory) -> None:
    actions = np.zeros((3, len(JOINTS)), dtype=np.float32)
    actions[:, -1] = 1.016
    policy = FakePolicy(
        actions,
        n_action_steps=2,
        action_lower=[-10.0, -10.0, -10.0, 0.0],
        action_upper=[10.0, 10.0, 10.0, 1.0],
    )
    module, control = make_runtime(policy)
    _provide_observation(module)

    _preflight(module)
    module.start_rollout()
    wait_until(lambda: control.execute_trajectory.call_count >= 1, timeout=1.0)
    module.stop_rollout()

    trajectory = control.execute_trajectory.call_args_list[0].args[0]
    assert [point.positions[-1] for point in trajectory.points[1:]] == [1.0, 1.0]


def test_next_chunk_uses_latest_joint_observation(make_runtime: RuntimeFactory) -> None:
    policy = FakePolicy(_action_chunk(), n_action_steps=1)
    module, control = make_runtime(policy)
    first_submitted = Event()
    release_first = Event()

    def execute_trajectory(*_args: object, **_kwargs: object) -> TrajectoryExecutionResult:
        if not first_submitted.is_set():
            first_submitted.set()
            assert release_first.wait(timeout=1.0)
        return TrajectoryExecutionResult(TrajectoryExecutionStatus.ACCEPTED)

    control.execute_trajectory.side_effect = execute_trajectory
    _provide_observation(module, positions=[0.0] * len(JOINTS))
    _preflight(module)
    module.start_rollout()
    assert first_submitted.wait(timeout=1.0)

    latest = [0.4, 0.3, 0.2, 0.1]
    _provide_observation(module, positions=latest)
    release_first.set()
    wait_until(lambda: control.execute_trajectory.call_count >= 2, timeout=1.0)
    module.stop_rollout()

    second = control.execute_trajectory.call_args_list[1].args[0]
    np.testing.assert_allclose(second.points[0].positions, latest)


def test_start_mismatch_waits_for_new_joint_state_before_retry(
    make_runtime: RuntimeFactory,
) -> None:
    policy = FakePolicy(_action_chunk(), n_action_steps=1)
    module, control = make_runtime(policy)
    _provide_observation(module)

    def execute_trajectory(*_args: object, **_kwargs: object) -> TrajectoryExecutionResult:
        status = (
            TrajectoryExecutionStatus.START_STATE_MISMATCH
            if control.execute_trajectory.call_count == 1
            else TrajectoryExecutionStatus.ACCEPTED
        )
        return TrajectoryExecutionResult(status)

    control.execute_trajectory.side_effect = execute_trajectory
    _preflight(module)
    module.start_rollout()
    wait_until(lambda: control.execute_trajectory.call_count == 1, timeout=1.0)

    _provide_observation(module, positions=[0.2] * len(JOINTS), ts=time.time() + 0.01)
    wait_until(lambda: control.execute_trajectory.call_count >= 2, timeout=1.0)
    module.stop_rollout()

    assert module.rollout_status()["last_error"] is None


def test_a_press_stops_worker_before_cancelling_its_trajectory(
    make_runtime: RuntimeFactory,
) -> None:
    policy = FakePolicy(_action_chunk(), n_action_steps=2)
    module, control = make_runtime(policy)
    _provide_observation(module)
    _preflight(module)
    execute_started = Event()
    release_execute = Event()
    stop_finished = Event()

    def execute_trajectory(*_args: object, **_kwargs: object) -> TrajectoryExecutionResult:
        execute_started.set()
        assert release_execute.wait(timeout=1.0)
        return TrajectoryExecutionResult(TrajectoryExecutionStatus.ACCEPTED)

    control.execute_trajectory.side_effect = execute_trajectory
    pressed = Buttons()
    pressed.right_primary = True

    def stop_from_button() -> None:
        module._on_button_pressed(pressed)
        stop_finished.set()

    module._on_button_pressed(pressed)
    assert execute_started.wait(timeout=1.0)
    stop_thread = Thread(target=stop_from_button)
    stop_thread.start()
    try:
        assert module._stop_event.wait(timeout=1.0)
        assert module.rollout_status()["active"] is True
        control.cancel_trajectory.assert_not_called()
    finally:
        release_execute.set()
        stop_thread.join(timeout=1.0)

    assert stop_finished.is_set()
    assert control.execute_trajectory.call_count == 1
    control.cancel_trajectory.assert_called_with()


def test_uncertain_cancellation_is_reported(make_runtime: RuntimeFactory) -> None:
    policy = FakePolicy(_action_chunk(), n_action_steps=1)
    module, control = make_runtime(policy)
    _provide_observation(module)
    _preflight(module)
    control.cancel_trajectory.return_value = TrajectoryCancellationResult(
        TrajectoryCancellationStatus.UNCERTAIN,
        "coordinator did not confirm cancellation",
    )

    module.start_rollout()
    wait_until(lambda: control.execute_trajectory.call_count >= 1, timeout=1.0)
    status = module.stop_rollout()

    assert status["active"] is False
    assert status["last_error"] == "coordinator did not confirm cancellation"


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        (np.zeros((3, len(JOINTS) - 1), dtype=np.float32), "action width"),
        (np.full((3, len(JOINTS)), np.nan, dtype=np.float32), "non-finite joint targets"),
    ],
)
def test_invalid_action_chunk_cancels_and_latches_rollout_off(
    make_runtime: RuntimeFactory,
    actions: NDArray[np.float32],
    message: str,
) -> None:
    policy = FakePolicy(actions)
    module, control = make_runtime(policy)
    _provide_observation(module)

    _preflight(module)
    module.start_rollout()

    wait_until(lambda: module.rollout_status()["active"] is False, timeout=1.0)
    assert message in (module.rollout_status()["last_error"] or "")
    control.cancel_trajectory.assert_called_with()


def test_trajectory_rejection_cancels_and_latches_rollout_off(
    make_runtime: RuntimeFactory,
) -> None:
    policy = FakePolicy(_action_chunk())
    module, control = make_runtime(policy)
    _provide_observation(module)
    control.execute_trajectory.return_value = TrajectoryExecutionResult(
        TrajectoryExecutionStatus.INVALID_TRAJECTORY,
        "outside hardware limits",
    )

    _preflight(module)
    module.start_rollout()

    wait_until(lambda: module.rollout_status()["active"] is False, timeout=1.0)
    assert module.rollout_status()["last_error"] == "outside hardware limits"
    control.cancel_trajectory.assert_called_with()


@pytest.mark.parametrize(
    ("configure", "message"),
    [
        (lambda config: setattr(config, "n_action_steps", None), "positive int"),
        (lambda config: setattr(config, "n_action_steps", 0), "positive int"),
        (
            lambda config: setattr(
                config.input_features["observation.images.wrist"],
                "shape",
                (3, 8, 8),
            ),
            "Policy image shape",
        ),
    ],
)
def test_incompatible_chunk_contract_is_rejected(
    make_runtime: RuntimeFactory,
    configure: Callable[[FakeUpstreamConfig], None],
    message: str,
) -> None:
    policy = FakePolicy(_action_chunk())
    configure(policy.upstream_config)
    module, control = make_runtime(policy)
    _provide_observation(module)

    status = module.preflight_rollout()

    assert status["active"] is False
    assert status["policy_ready"] is False
    assert message in (status["last_error"] or "")
    control.execute_trajectory.assert_not_called()


def test_policy_refuses_to_load_without_live_observations(make_runtime: RuntimeFactory) -> None:
    policy = FakePolicy(_action_chunk())
    module, control = make_runtime(policy)

    result = module.preflight_rollout()

    assert result["active"] is False
    assert "no camera image" in (result["last_error"] or "")
    assert policy.config_load_count == 0
    control.execute_trajectory.assert_not_called()


def test_policy_refuses_stale_observations(make_runtime: RuntimeFactory) -> None:
    policy = FakePolicy(_action_chunk())
    module, control = make_runtime(policy)
    _provide_observation(module, ts=time.time() - module.config.max_observation_age_s - 1.0)

    result = module.preflight_rollout()

    assert result["active"] is False
    assert "camera image is stale" in (result["last_error"] or "")
    assert policy.config_load_count == 0
    control.execute_trajectory.assert_not_called()


def test_checkpoint_loads_on_demand_and_is_cached(make_runtime: RuntimeFactory) -> None:
    policy = FakePolicy(_action_chunk())
    module, control = make_runtime(policy)
    _provide_observation(module)

    _preflight(module)
    module.start_rollout()
    wait_until(lambda: control.execute_trajectory.call_count >= 1, timeout=1.0)
    module.stop_rollout()
    control.execute_trajectory.reset_mock()
    module.start_rollout()
    wait_until(lambda: control.execute_trajectory.call_count >= 1, timeout=1.0)
    module.stop_rollout()

    assert policy.config_load_count == 1


def test_start_requires_a_successful_preflight(make_runtime: RuntimeFactory) -> None:
    policy = FakePolicy(_action_chunk())
    module, control = make_runtime(policy)
    _provide_observation(module)

    status = module.start_rollout()

    assert status["active"] is False
    assert status["last_error"] == "policy preflight has not passed"
    control.execute_trajectory.assert_not_called()


def test_preflight_never_sends_a_trajectory(make_runtime: RuntimeFactory) -> None:
    policy = FakePolicy(_action_chunk())
    module, control = make_runtime(policy)
    _provide_observation(module)

    status = module.preflight_rollout()

    assert status["policy_ready"] is True
    control.execute_trajectory.assert_not_called()
    control.cancel_trajectory.assert_not_called()


def test_preflight_requires_the_configured_coordinator_task(
    make_runtime: RuntimeFactory,
) -> None:
    policy = FakePolicy(_action_chunk())
    module, control = make_runtime(policy)
    _provide_observation(module)
    control.list_tasks.return_value = ["another_task"]

    status = module.preflight_rollout()

    assert status["policy_ready"] is False
    assert "missing trajectory task" in (status["last_error"] or "")
    assert policy.config_load_count == 0


@pytest.mark.parametrize(
    ("positions", "names", "message"),
    [
        ([0.0, 0.0, 0.0, float("nan")], JOINTS, "non-finite positions"),
        ([0.0, 0.0, 0.0], JOINTS[:-1], "missing configured joints"),
    ],
)
def test_preflight_rejects_invalid_live_joints(
    make_runtime: RuntimeFactory,
    positions: list[float],
    names: list[str],
    message: str,
) -> None:
    policy = FakePolicy(_action_chunk())
    module, control = make_runtime(policy)
    timestamp = time.time()
    module._on_color_image(
        Image(data=np.zeros((4, 5, 3), dtype=np.uint8), format=ImageFormat.RGB, ts=timestamp)
    )
    module._on_joint_state(JointState(ts=timestamp, name=names, position=positions))

    status = module.preflight_rollout()

    assert status["policy_ready"] is False
    assert message in (status["last_error"] or "")
    control.execute_trajectory.assert_not_called()


@pytest.mark.parametrize(
    ("image", "image_format"),
    [
        (np.zeros((8, 8, 3), dtype=np.uint8), ImageFormat.RGB),
        (np.zeros((4, 5, 3), dtype=np.uint8), ImageFormat.BGR),
    ],
)
def test_preflight_requires_exact_live_rgb_contract(
    make_runtime: RuntimeFactory,
    image: NDArray[np.uint8],
    image_format: ImageFormat,
) -> None:
    policy = FakePolicy(_action_chunk())
    module, control = make_runtime(policy)
    timestamp = time.time()
    module._on_color_image(Image(data=image, format=image_format, ts=timestamp))
    module._on_joint_state(JointState(ts=timestamp, name=JOINTS, position=[0.0] * len(JOINTS)))

    status = module.preflight_rollout()

    assert status["policy_ready"] is False
    assert "no camera image" in (status["last_error"] or "")
    control.execute_trajectory.assert_not_called()


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        ([-1.0] * 3, [1.0] * 3, "range shape"),
        ([-1.0, -1.0, -1.0, float("nan")], [1.0] * 4, "non-finite"),
        ([2.0] * 4, [1.0] * 4, "min greater than max"),
    ],
)
def test_preflight_rejects_invalid_checkpoint_action_bounds(
    make_runtime: RuntimeFactory,
    lower: list[float],
    upper: list[float],
    message: str,
) -> None:
    policy = FakePolicy(_action_chunk(), action_lower=lower, action_upper=upper)
    module, control = make_runtime(policy)
    _provide_observation(module)

    status = module.preflight_rollout()

    assert status["policy_ready"] is False
    assert message in (status["last_error"] or "")
    control.execute_trajectory.assert_not_called()


def test_preflight_rejects_unavailable_cuda(
    make_runtime: RuntimeFactory,
    mocker: pytest_mock.MockerFixture,
) -> None:
    policy = FakePolicy(_action_chunk())
    module, control = make_runtime(policy, device="cuda")
    _provide_observation(module)
    mocker.patch.object(torch.cuda, "is_available", return_value=False)

    status = module.preflight_rollout()

    assert status["policy_ready"] is False
    assert "CUDA is not available" in (status["last_error"] or "")
    control.execute_trajectory.assert_not_called()


@pytest.mark.parametrize("grip", ["left_grip", "right_grip"])
def test_grip_takeover_stops_rollout_until_explicit_restart(
    make_runtime: RuntimeFactory, grip: str
) -> None:
    module, control = make_runtime(FakePolicy(_action_chunk()))
    _provide_observation(module)
    _preflight(module)
    module.start_rollout()
    wait_until(lambda: control.execute_trajectory.call_count > 0, timeout=1)
    held = Buttons()
    setattr(held, grip, True)

    module._on_teleop_buttons(held)
    wait_until(lambda: not module.rollout_status()["active"], timeout=1)
    assert module.start_rollout()["active"] is False
    assert "release" in (module.rollout_status()["last_error"] or "")
    submissions = control.execute_trajectory.call_count
    module._on_teleop_buttons(Buttons())
    assert module.rollout_status()["active"] is False
    assert control.execute_trajectory.call_count == submissions
    _provide_observation(module)
    assert module.start_rollout()["active"] is True
    module.stop_rollout()


def test_grip_during_inference_discards_result_without_blocking_input(
    make_runtime: RuntimeFactory, mocker: pytest_mock.MockerFixture
) -> None:
    module, control = make_runtime(FakePolicy(_action_chunk()))
    _provide_observation(module)
    _preflight(module)
    predicting = Event()
    release_prediction = Event()

    def predict(*args: Any, **kwargs: Any) -> NDArray[np.float32]:
        predicting.set()
        assert release_prediction.wait(timeout=2)
        return _action_chunk()[None, :]

    mocker.patch.object(module, "_predict", side_effect=predict)
    module.start_rollout()
    try:
        assert predicting.wait(timeout=1)
        held = Buttons()
        held.right_grip = True
        module._on_teleop_buttons(held)
        module._on_teleop_buttons(Buttons())
        assert module._stop_event.is_set()
        control.execute_trajectory.assert_not_called()
    finally:
        release_prediction.set()
        module.stop_rollout()
    control.execute_trajectory.assert_not_called()
    assert module.rollout_status()["active"] is False


def test_stopping_inactive_policy_leaves_planner_trajectory_alone(
    make_runtime: RuntimeFactory,
) -> None:
    module, control = make_runtime(FakePolicy(_action_chunk()))
    module.stop_rollout()
    held = Buttons()
    held.right_grip = True
    module._on_teleop_buttons(held)
    control.cancel_trajectory.assert_not_called()


class SequencedPolicy(FakePolicy):
    """A fake policy that answers successive predictions with successive chunks."""

    def __init__(self, chunks: list[NDArray[np.float32]], n_action_steps: int) -> None:
        super().__init__(chunks[0], n_action_steps=n_action_steps)
        self.chunks = [torch.from_numpy(c).unsqueeze(0) for c in chunks]
        self.predictions = 0

    def predict_action_chunk(self, batch: dict[str, object]) -> Tensor:
        self.batch = dict(batch)
        chunk = self.chunks[min(self.predictions, len(self.chunks) - 1)]
        self.predictions += 1
        self.called.set()
        return chunk


def _constant_chunk(values: list[float]) -> NDArray[np.float32]:
    return np.array([[v] * len(JOINTS) for v in values], dtype=np.float32)


def test_action_ensemble_weighs_the_oldest_chunk_most() -> None:
    ensemble = _ActionEnsemble(coeff=np.log(2.0))  # weights 1, 1/2, 1/4 ...
    ensemble.add(0, _constant_chunk([0.0, 0.0, 0.0]))
    ensemble.add(1, _constant_chunk([3.0, 3.0, 3.0]))
    ensemble.add(2, _constant_chunk([7.0, 7.0, 7.0]))

    targets = ensemble.targets(2, 4)

    assert targets.shape == (3, len(JOINTS))  # nothing predicted step 5
    # step 2: (1 * 0 + 1/2 * 3 + 1/4 * 7) / (1 + 1/2 + 1/4)
    np.testing.assert_allclose(targets[0], (1.5 + 1.75) / 1.75)
    # step 3: (1 * 3 + 1/2 * 7) / 1.5
    np.testing.assert_allclose(targets[1], 6.5 / 1.5)
    np.testing.assert_allclose(targets[2], 7.0)


def test_action_ensemble_drops_expired_chunks_and_can_discard_the_last() -> None:
    ensemble = _ActionEnsemble(coeff=0.0)
    ensemble.add(0, _constant_chunk([1.0, 1.0]))
    ensemble.add(2, _constant_chunk([5.0, 5.0]))  # the first chunk ends before step 2
    np.testing.assert_allclose(ensemble.targets(2, 2), 5.0)
    ensemble.add(3, _constant_chunk([9.0, 9.0]))
    ensemble.discard_last()
    np.testing.assert_allclose(ensemble.targets(3, 1), 5.0)
    with pytest.raises(ValueError, match="step order"):
        ensemble.add(1, _constant_chunk([0.0]))


def test_action_ensemble_window_keeps_only_the_newest_chunks() -> None:
    ensemble = _ActionEnsemble(coeff=0.0, window=2)
    ensemble.add(0, _constant_chunk([0.0, 0.0, 0.0, 0.0]))
    ensemble.add(1, _constant_chunk([4.0, 4.0, 4.0, 4.0]))
    ensemble.add(2, _constant_chunk([8.0, 8.0, 8.0, 8.0]))
    # The oldest chunk still covers step 2 but fell out of the window.
    np.testing.assert_allclose(ensemble.targets(2, 1), 6.0)


def test_action_ensemble_without_coefficient_returns_the_newest_chunk() -> None:
    ensemble = _ActionEnsemble(coeff=None)
    ensemble.add(0, _constant_chunk([0.0, 0.0, 0.0]))
    ensemble.add(1, _constant_chunk([3.0, 3.0, 3.0]))
    np.testing.assert_allclose(ensemble.targets(1, 2), 3.0)


@pytest.mark.parametrize(
    ("coeff", "expected_first", "expected_second"),
    [(0.0, 5.5, 6.5), (None, 10.0, 11.0)],
)
def test_second_submission_ensembles_the_overlapping_chunks(
    make_runtime: RuntimeFactory,
    coeff: float | None,
    expected_first: float,
    expected_second: float,
) -> None:
    policy = SequencedPolicy(
        [_constant_chunk([0.0, 1.0, 2.0]), _constant_chunk([10.0, 11.0, 12.0])],
        n_action_steps=2,
    )
    module, control = make_runtime(policy, temporal_ensemble_coeff=coeff)
    _provide_observation(module)
    _preflight(module)
    module.start_rollout()
    wait_until(lambda: control.execute_trajectory.call_count >= 2, timeout=1.0)
    module.stop_rollout()

    first = control.execute_trajectory.call_args_list[0].args[0]
    second = control.execute_trajectory.call_args_list[1].args[0]
    np.testing.assert_allclose([p.positions for p in first.points[1:]], [[0.0] * 4, [1.0] * 4])
    assert [p.time_from_start for p in second.points] == [0.0, 0.02, 0.04]
    # step 1 was predicted by chunk 0 (index 1) and chunk 1 (index 0), step 2 likewise.
    np.testing.assert_allclose(second.points[1].positions, expected_first)
    np.testing.assert_allclose(second.points[2].positions, expected_second)


def test_replan_steps_overlap_the_running_trajectory(
    make_runtime: RuntimeFactory, tmp_path: Path
) -> None:
    policy = FakePolicy(_action_chunk(), n_action_steps=3)
    module, control = make_runtime(
        policy, replan_steps=2, rollout_log_dir=str(tmp_path), temporal_ensemble_coeff=None
    )
    _provide_observation(module)
    _preflight(module)
    log_path = Path(module.start_rollout()["rollout_log"] or "")
    wait_until(lambda: control.execute_trajectory.call_count >= 3, timeout=2.0)
    module.stop_rollout()
    wait_until(lambda: module.rollout_status()["rollout_log"] is None, timeout=1.0)

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    chunks = [r for r in records if r["type"] == "chunk"][:3]
    assert [c["step"] for c in chunks] == [0, 2, 4]
    assert all(c["executed_steps"] == 2 and len(c["sent"]) == 3 for c in chunks)
    # Each full 3-step trajectory is replaced after 2 steps at 50 fps.
    periods = np.diff([c["sent_t"] for c in chunks])
    assert np.all(periods >= 0.03) and np.all(periods < 0.3)
    for call in control.execute_trajectory.call_args_list[:3]:
        assert len(call.args[0].points) == 4


def test_late_submission_skips_the_steps_the_running_trajectory_covered(
    make_runtime: RuntimeFactory, tmp_path: Path
) -> None:
    policy = FakePolicy(_action_chunk(), n_action_steps=3)
    module, control = make_runtime(policy, fps=10.0, rollout_log_dir=str(tmp_path))
    first = Event()

    def execute_trajectory(*_args: object, **_kwargs: object) -> TrajectoryExecutionResult:
        if not first.is_set():
            first.set()
            time.sleep(0.25)  # one 100 ms step plus a bit of a late RPC
        return TrajectoryExecutionResult(TrajectoryExecutionStatus.ACCEPTED)

    control.execute_trajectory.side_effect = execute_trajectory
    _provide_observation(module)
    _preflight(module)
    log_path = Path(module.start_rollout()["rollout_log"] or "")
    wait_until(lambda: control.execute_trajectory.call_count >= 2, timeout=2.0)
    module.stop_rollout()
    wait_until(lambda: module.rollout_status()["rollout_log"] is None, timeout=1.0)

    steps = [
        r["step"]
        for r in map(json.loads, log_path.read_text().splitlines())
        if r["type"] == "chunk"
    ]
    assert steps[:2] == [0, 2]


def test_replan_steps_none_predicts_once_per_chunk(
    make_runtime: RuntimeFactory, tmp_path: Path
) -> None:
    policy = FakePolicy(_action_chunk(), n_action_steps=2)
    module, control = make_runtime(
        policy, replan_steps=None, temporal_ensemble_coeff=None, rollout_log_dir=str(tmp_path)
    )
    _provide_observation(module)
    _preflight(module)
    log_path = Path(module.start_rollout()["rollout_log"] or "")
    wait_until(lambda: control.execute_trajectory.call_count >= 3, timeout=2.0)
    module.stop_rollout()
    wait_until(lambda: module.rollout_status()["rollout_log"] is None, timeout=1.0)

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    chunks = [r for r in records if r["type"] == "chunk"][:3]
    assert [c["step"] for c in chunks] == [0, 2, 4]
    assert all(c["executed_steps"] == 2 for c in chunks)
    assert records[0]["replan_steps"] is None


def test_replan_steps_beyond_the_checkpoint_horizon_is_rejected(
    make_runtime: RuntimeFactory,
) -> None:
    module, control = make_runtime(FakePolicy(_action_chunk(), n_action_steps=2), replan_steps=3)
    _provide_observation(module)

    status = module.preflight_rollout()

    assert status["policy_ready"] is False
    assert "replan_steps 3 exceeds" in (status["last_error"] or "")
    control.execute_trajectory.assert_not_called()
