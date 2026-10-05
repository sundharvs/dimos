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

"""Run trained LeRobot policies in an isolated Python environment."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Condition, Event, RLock, Thread, current_thread
import time
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction, RobotObservation
from lerobot.utils.import_utils import register_third_party_plugins
import numpy as np
from numpy.typing import NDArray
from reactivex.disposable import Disposable
import torch

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.control.tasks.trajectory_task.trajectory_task import (
    JOINT_TRAJECTORY_TASK_NAME,
    TrajectoryExecutionStatus,
)
from dimos.core.core import rpc
from dimos.imitation.policy.lerobot.module import (
    LeRobotPolicyModule,
    RolloutStatus,
)
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Float32 import Float32
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.teleop.webxr.controller_types import BUTTON_ALIASES, Buttons
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_STATE_FEATURE = "observation.state"
_ACTION_FEATURE = "action"

RawObservation = dict[str, NDArray[np.uint8] | NDArray[np.float32]]

# Gripper openings closer than this to the last published one are not re-sent.
_GRIPPER_PUBLISH_TOLERANCE = 0.005


class _ActionEnsemble:
    """Temporal ensembling over overlapping action chunks (ACT, Algorithm 2).

    Chunks are registered with the step their first action targets. The target
    for a step is the weighted mean of every registered chunk's prediction for
    it, weights ``exp(-coeff * i)`` with ``i = 0`` for the oldest chunk, as in
    LeRobot's ``ACTTemporalEnsembler``. ``coeff=None`` disables averaging and
    returns the newest chunk's prediction. ``window`` keeps only the newest
    that many chunks.
    """

    def __init__(self, coeff: float | None, window: int | None = None) -> None:
        self._coeff = coeff
        self._window = window
        self._chunks: list[tuple[int, NDArray[np.float32]]] = []

    def add(self, step: int, chunk: NDArray[np.float32]) -> None:
        """Register ``chunk`` (steps, width) whose first row targets ``step``."""
        if self._chunks and step < self._chunks[-1][0]:
            raise ValueError("chunks must be added in step order")
        self._chunks = [(start, rows) for start, rows in self._chunks if start + len(rows) > step]
        self._chunks.append((step, chunk))
        if self._window is not None:
            del self._chunks[: -self._window]

    def discard_last(self) -> None:
        if self._chunks:
            self._chunks.pop()

    def targets(self, step: int, horizon: int) -> NDArray[np.float32]:
        """Targets for ``step .. step + horizon - 1``, fewer if no chunk reaches that far."""
        rows: list[NDArray[np.float32]] = []
        for target_step in range(step, step + horizon):
            predictions = [
                chunk[target_step - start]
                for start, chunk in self._chunks
                if start <= target_step < start + len(chunk)
            ]
            if not predictions:
                break
            if self._coeff is None:
                rows.append(predictions[-1])
                continue
            weights = np.exp(-self._coeff * np.arange(len(predictions), dtype=np.float64))
            stacked = np.asarray(predictions, dtype=np.float64)
            rows.append(
                ((weights[:, None] * stacked).sum(axis=0) / weights.sum()).astype(np.float32)
            )
        return np.asarray(rows, dtype=np.float32).reshape(len(rows), -1)


@dataclass(frozen=True)
class _LoadedPolicy:
    policy: PreTrainedPolicy
    device: torch.device
    preprocessor: PolicyProcessorPipeline[RobotObservation, RobotObservation]
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction]
    use_amp: bool
    chunk_size: int | None
    n_action_steps: int
    action_lower: NDArray[np.float32]
    action_upper: NDArray[np.float32]


class _RolloutLog:
    """JSON-lines record of one rollout: header, joint states, predicted chunks, end."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("w", buffering=1)
        self._lock = RLock()

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            if self._file.closed:
                return
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


class LeRobotPolicyRuntime(LeRobotPolicyModule):
    """Concrete LeRobot implementation loaded by ``LeRobotPolicyModule``."""

    _lock: RLock
    _observation_changed: Condition
    _loaded_policy: _LoadedPolicy | None
    _latest_image: tuple[NDArray[np.uint8], float] | None
    _latest_joint_state: JointState | None
    _stop_event: Event
    _thread: Thread | None
    _chunks_accepted: int
    _last_error: str | None
    _active: bool

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = RLock()
        self._observation_changed = Condition(self._lock)
        self._loaded_policy = None
        self._latest_image = None
        self._latest_joint_state = None
        self._stop_event = Event()
        self._thread = None
        self._chunks_accepted = 0
        self._last_error = None
        self._active = False
        self._manual_control = False
        self._log: _RolloutLog | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color_image)))
        self.register_disposable(
            Disposable(self.coordinator_joint_state.subscribe(self._on_joint_state))
        )
        self.register_disposable(Disposable(self.button_pressed.subscribe(self._on_button_pressed)))
        self.register_disposable(Disposable(self.teleop_buttons.subscribe(self._on_teleop_buttons)))

    @rpc
    def stop(self) -> None:
        if not self._stop_policy():
            self._cancel_after_stop_timeout()
        super().stop()

    @rpc
    def preflight_rollout(self) -> RolloutStatus:
        """Validate the checkpoint, coordinator, and live observations without moving."""
        with self._lock:
            if self._active:
                self._last_error = "cannot preflight while a policy rollout is active"
                return self._status_locked()
            loaded_policy = self._loaded_policy
            try:
                self._snapshot_observation(time.time())
            except Exception as exc:
                self._loaded_policy = None
                self._last_error = str(exc)
                return self._status_locked()

        try:
            tasks = set(self._control.list_tasks())
            if JOINT_TRAJECTORY_TASK_NAME not in tasks:
                raise RuntimeError(
                    f"ControlCoordinator is missing trajectory task {JOINT_TRAJECTORY_TASK_NAME!r}"
                )
            if loaded_policy is None:
                loaded_policy = self._load_policy()
                logger.info(
                    "Loaded LeRobot policy during preflight",
                    path=self.config.policy_path,
                    runtime_fps=self.config.fps,
                    chunk_size=loaded_policy.chunk_size,
                    n_action_steps=loaded_policy.n_action_steps,
                )
            with self._lock:
                self._loaded_policy = loaded_policy
                self._snapshot_observation(time.time())
                self._last_error = None
                return self._status_locked()
        except Exception as exc:
            with self._lock:
                self._loaded_policy = None
                self._last_error = str(exc)
                return self._status_locked()

    @rpc
    def start_rollout(self) -> RolloutStatus:
        with self._lock:
            if self._manual_control:
                self._last_error = "release the controller grips before starting rollout"
                return self._status_locked()
            if self._thread is not None and self._thread.is_alive():
                self._last_error = "a policy rollout is already active"
                return self._status_locked()
            if self._loaded_policy is None:
                self._last_error = "policy preflight has not passed"
                return self._status_locked()
            try:
                self._snapshot_observation(time.time())
            except RuntimeError as exc:
                self._last_error = str(exc)
                return self._status_locked()

            self._stop_event.clear()
            self._chunks_accepted = 0
            self._last_error = None
            self._active = True
            self._open_log()
            self._thread = Thread(
                target=self._run_rollout,
                name="lerobot-policy-rollout",
                daemon=True,
            )
            self._thread.start()
            return self._status_locked()

    @rpc
    def stop_rollout(self) -> RolloutStatus:
        if not self._stop_policy():
            self._cancel_after_stop_timeout()
        return self.rollout_status()

    @rpc
    def rollout_status(self) -> RolloutStatus:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> RolloutStatus:
        try:
            self._snapshot_observation(time.time())
            observations_ready = True
        except RuntimeError:
            observations_ready = False
        return {
            "active": self._active,
            "policy_path": self.config.policy_path,
            "task": self.config.task,
            "device": self.config.device,
            "policy_ready": self._loaded_policy is not None,
            "observations_ready": observations_ready,
            "chunks_accepted": self._chunks_accepted,
            "rollout_log": str(self._log.path) if self._log is not None else None,
            "last_error": self._last_error,
        }

    def _on_color_image(self, image: Image) -> None:
        if image.format != ImageFormat.RGB or image.data.dtype != np.uint8:
            logger.warning("Ignoring non-uint8 RGB policy image", image=str(image))
            return
        expected_shape = (self.config.image_height, self.config.image_width, 3)
        if image.data.shape != expected_shape:
            logger.warning(
                "Ignoring policy image with unexpected shape",
                shape=image.data.shape,
                expected=expected_shape,
            )
            return
        with self._lock:
            self._latest_image = (np.ascontiguousarray(image.data), image.ts)

    def _on_joint_state(self, state: JointState) -> None:
        with self._observation_changed:
            self._latest_joint_state = JointState(state)
            log = self._log
            self._observation_changed.notify_all()
        if log is not None:
            positions = dict(zip(state.name, state.position, strict=False))
            log.write(
                {
                    "type": "joint_state",
                    "t": float(state.ts),
                    "pos": [
                        float(positions.get(name, float("nan"))) for name in self.config.joint_names
                    ],
                }
            )

    def _on_teleop_buttons(self, buttons: Buttons) -> None:
        with self._observation_changed:
            self._manual_control = buttons.left_grip or buttons.right_grip
            if self._manual_control and self._active:
                self._stop_event.set()
                self._observation_changed.notify_all()

    def _on_button_pressed(self, buttons: Buttons) -> None:
        button = BUTTON_ALIASES.get(self.config.rollout_button, self.config.rollout_button)
        if not bool(getattr(buttons, button)):
            return
        with self._lock:
            active = self._active
        if active:
            self.stop_rollout()
        else:
            self.start_rollout()

    def _snapshot_observation(
        self, now: float
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32], float]:
        if self._latest_image is None:
            raise RuntimeError("no camera image has been received")
        if self._latest_joint_state is None:
            raise RuntimeError("no coordinator joint state has been received")

        image, image_ts = self._latest_image
        state = self._latest_joint_state
        max_age = self.config.max_observation_age_s
        if now - image_ts > max_age:
            raise RuntimeError(f"camera image is stale by {now - image_ts:.2f}s")
        if now - state.ts > max_age:
            raise RuntimeError(f"joint state is stale by {now - state.ts:.2f}s")

        positions = dict(zip(state.name, state.position, strict=False))
        missing = [name for name in self.config.joint_names if name not in positions]
        if missing:
            raise RuntimeError(f"joint state is missing configured joints: {missing}")
        vector = np.asarray(
            [positions[name] for name in self.config.joint_names],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(vector)):
            raise RuntimeError("joint state contains non-finite positions")
        return image.copy(), vector, state.ts

    def _load_policy(self) -> _LoadedPolicy:
        register_third_party_plugins()
        policy_config = PreTrainedConfig.from_pretrained(self.config.policy_path)
        if self.config.device is not None:
            policy_config.device = self.config.device
        if policy_config.device is None:
            raise RuntimeError("LeRobot did not resolve an inference device")

        self._validate_features(policy_config)
        device = torch.device(policy_config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Policy requested device {policy_config.device!r}, but CUDA is not available"
            )

        policy_class = get_policy_class(policy_config.type)
        loaded_policy = policy_class.from_pretrained(self.config.policy_path, config=policy_config)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=self.config.policy_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        action_lower, action_upper = _checkpoint_action_bounds(
            postprocessor,
            len(self.config.joint_names),
        )
        n_action_steps = _positive_int_attribute(policy_config, "n_action_steps")
        if self.config.replan_steps is not None and self.config.replan_steps > n_action_steps:
            raise ValueError(
                f"replan_steps {self.config.replan_steps} exceeds the checkpoint's "
                f"n_action_steps {n_action_steps}"
            )
        return _LoadedPolicy(
            policy=loaded_policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=bool(policy_config.use_amp),
            chunk_size=_optional_int_attribute(policy_config, "chunk_size"),
            n_action_steps=n_action_steps,
            action_lower=action_lower,
            action_upper=action_upper,
        )

    def _validate_features(self, policy_config: PreTrainedConfig) -> None:
        inputs = policy_config.input_features or {}
        outputs = policy_config.output_features or {}
        image_feature = self.config.image_feature
        missing = {image_feature, _STATE_FEATURE} - set(inputs)
        if missing:
            raise ValueError(
                "Policy is incompatible with the DimOS single-camera runtime; "
                f"missing input features: {sorted(missing)}"
            )
        if _ACTION_FEATURE not in outputs:
            raise ValueError(f"Policy has no {_ACTION_FEATURE!r} output feature")
        state_shape = tuple(inputs[_STATE_FEATURE].shape)
        image_shape = tuple(inputs[image_feature].shape)
        action_shape = tuple(outputs[_ACTION_FEATURE].shape)
        joint_count = len(self.config.joint_names)
        expected_image_shape = (3, self.config.image_height, self.config.image_width)
        if image_shape != expected_image_shape:
            raise ValueError(
                f"Policy image shape {image_shape} does not match {expected_image_shape}"
            )
        if not state_shape or state_shape[0] != joint_count:
            raise ValueError(
                f"Policy state dimension {state_shape} does not match {joint_count} configured joints"
            )
        if not action_shape or action_shape[0] != joint_count:
            raise ValueError(
                f"Policy action dimension {action_shape} does not match {joint_count} configured joints"
            )

    def _predict(
        self,
        loaded_policy: _LoadedPolicy,
        image: NDArray[np.uint8],
        state: NDArray[np.float32],
        *,
        task: str,
    ) -> NDArray[np.float32]:
        observation: RawObservation = {
            self.config.image_feature: image,
            _STATE_FEATURE: state,
        }
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda")
            if loaded_policy.device.type == "cuda" and loaded_policy.use_amp
            else nullcontext(),
        ):
            prepared = prepare_observation_for_inference(
                observation,
                loaded_policy.device,
                task=task,
                robot_type=self.config.robot_type,
            )
            prepared = loaded_policy.preprocessor(prepared)
            predict = getattr(loaded_policy.policy, "predict_action_chunk", None)
            if not callable(predict):
                raise TypeError("Policy does not provide predict_action_chunk()")
            action_chunk = loaded_policy.postprocessor(predict(prepared))
        return np.asarray(action_chunk.to("cpu").numpy(), dtype=np.float32)

    def _run_rollout(self) -> None:
        loaded_policy: _LoadedPolicy | None = None
        try:
            with self._lock:
                loaded_policy = self._loaded_policy
            if loaded_policy is None:
                raise RuntimeError("policy preflight has not passed")
            if self._stop_event.is_set():
                return
            self._reset_policy(loaded_policy)
            self._run_steps(loaded_policy)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            logger.exception("LeRobot policy execution stopped", error=str(exc))
        finally:
            self._stop_event.set()
            cancellation_error = self._cancel_trajectory()
            self._close_log(cancellation_error)
            if loaded_policy is not None:
                self._reset_policy(loaded_policy)
            with self._lock:
                if cancellation_error is not None:
                    self._last_error = (
                        f"{self._last_error}; {cancellation_error}"
                        if self._last_error is not None
                        else cancellation_error
                    )
                self._active = False

    def _run_steps(self, loaded_policy: _LoadedPolicy) -> None:
        """Predict every ``replan_steps`` steps and keep the arm on an ensembled trajectory.

        Step ``k`` nominally begins ``k / fps`` after the first submission. Each
        iteration predicts a chunk for the step about to begin, folds it into
        the temporal ensemble, and submits the ensemble's next ``n_action_steps``
        targets timed so the trajectory lands as that step begins. The previous
        trajectory is still running then, so the coordinator anchors the new one
        to its commanded position and the arm never stops. Should a submission
        slip by whole steps, the previous trajectory's remaining targets cover
        them and the step counter skips ahead.
        """
        period = 1.0 / self.config.fps
        horizon = loaded_policy.n_action_steps
        replan = self.config.replan_steps or horizon
        ensemble = _ActionEnsemble(self.config.temporal_ensemble_coeff, self.config.ensemble_window)
        gripper_index = self._gripper_index()
        arm = [i for i in range(len(self.config.joint_names)) if i != gripper_index]
        step = 0
        clock0: float | None = None  # wall time at which step 0 began
        lead = 0.0  # prediction + submission latency of the previous iteration
        rpc_s = 0.0  # submission latency of the previous iteration
        last_opening: float | None = None
        while not self._stop_event.is_set():
            with self._lock:
                image, state, state_ts = self._snapshot_observation(time.time())
            t_predict = time.time()
            raw_chunk = self._predict(loaded_policy, image, state, task=self.config.task)
            infer_s = time.time() - t_predict
            chunk, clipped = self._bounded_chunk(loaded_policy, raw_chunk)
            if self._stop_event.is_set():
                break
            if clock0 is not None:
                # Steps the running trajectory will have passed by the time this one lands.
                slip = int(max(0.0, time.time() + rpc_s - (clock0 + step * period)) // period)
                if slip:
                    logger.warning("Policy submission late, skipping steps", steps=slip, step=step)
                    step += slip
            ensemble.add(step, chunk)
            targets = ensemble.targets(step, horizon)
            record: dict[str, Any] = {
                "type": "chunk",
                "t": t_predict,
                "step": step,
                "state_ts": float(state_ts),
                "state": [float(v) for v in state],
                "chunk": [[float(v) for v in row] for row in chunk],
                "executed_steps": replan,
                "clipped": [bool(v) for v in clipped],
                "infer_s": infer_s,
            }
            if gripper_index is not None:
                # The gripper task owns that joint: it gets the newest chunk's
                # first normalized opening (a command, not something to average)
                # and stays out of the arm trajectory.
                opening = float(np.clip(chunk[0, gripper_index], 0.0, 1.0))
                if last_opening is None or abs(opening - last_opening) > _GRIPPER_PUBLISH_TOLERANCE:
                    self.gripper_command.publish(Float32(data=opening))
                    last_opening = opening
                record["gripper_opening"] = opening
                targets[:, gripper_index] = opening
            record["sent"] = [[float(v) for v in row] for row in targets]
            result = self._control.execute_trajectory(self._trajectory(state[arm], targets[:, arm]))
            sent_t = time.time()
            rpc_s = sent_t - t_predict - infer_s
            record["result"] = result.status.name
            record["sent_t"] = sent_t
            if self._log is not None:
                self._log.write(record)
            if result.status is TrajectoryExecutionStatus.START_STATE_MISMATCH:
                ensemble.discard_last()
                self._wait_for_newer_joint_state(state_ts)
                continue
            if result.status is not TrajectoryExecutionStatus.ACCEPTED:
                raise RuntimeError(result.message or f"trajectory rejected: {result.status.name}")
            with self._lock:
                self._chunks_accepted += 1
            # A slow first inference (CUDA warm-up) must not pull the next step forward.
            lead = min(sent_t - t_predict, period)
            if clock0 is None:
                clock0 = sent_t
            step += replan
            self._stop_event.wait(max(0.0, clock0 + step * period - lead - time.time()))

    def _bounded_chunk(
        self, loaded_policy: _LoadedPolicy, action_chunk: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """Validate a predicted ``(1, steps, width)`` chunk and clip it to the checkpoint range."""
        expected_width = len(self.config.joint_names)
        if action_chunk.ndim != 3 or action_chunk.shape[0] != 1:
            raise RuntimeError(
                f"policy returned action chunk shape {action_chunk.shape}, expected "
                f"(1, steps, {expected_width})"
            )
        if action_chunk.shape[2] != expected_width:
            raise RuntimeError(
                f"policy returned action width {action_chunk.shape[2]}, expected {expected_width}"
            )
        if action_chunk.shape[1] < loaded_policy.n_action_steps:
            raise RuntimeError(
                f"policy returned {action_chunk.shape[1]} action steps, but n_action_steps "
                f"is {loaded_policy.n_action_steps}"
            )
        actions = action_chunk[0]
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("policy returned non-finite joint targets")
        bounded = np.clip(actions, loaded_policy.action_lower, loaded_policy.action_upper)
        clipped = np.any(actions != bounded, axis=0)
        if np.any(clipped):
            logger.warning(
                "Clipped policy actions to checkpoint range",
                joints=[
                    name
                    for name, was_clipped in zip(self.config.joint_names, clipped, strict=True)
                    if was_clipped
                ],
            )
        return np.array(bounded, dtype=np.float32), clipped

    def _open_log(self) -> None:
        """Start a JSONL log for this rollout (caller holds the lock)."""
        if self.config.rollout_log_dir is None:
            return
        directory = Path(self.config.rollout_log_dir).expanduser()
        path = directory / time.strftime("rollout_%Y%m%d_%H%M%S.jsonl")
        try:
            log = _RolloutLog(path)
        except OSError as exc:
            logger.warning("Rollout log disabled", path=str(path), error=str(exc))
            return
        loaded = self._loaded_policy
        log.write(
            {
                "type": "header",
                "t": time.time(),
                "policy_path": self.config.policy_path,
                "task": self.config.task,
                "joint_names": list(self.config.joint_names),
                "gripper_joint": self.config.gripper_joint,
                "image_feature": self.config.image_feature,
                "fps": self.config.fps,
                "chunk_size": loaded.chunk_size if loaded is not None else None,
                "n_action_steps": loaded.n_action_steps if loaded is not None else None,
                "replan_steps": self.config.replan_steps,
                "temporal_ensemble_coeff": self.config.temporal_ensemble_coeff,
                "ensemble_window": self.config.ensemble_window,
                "label": self.config.label,
            }
        )
        self._log = log
        logger.info("Rollout log opened", path=str(path))

    def _close_log(self, cancellation_error: str | None) -> None:
        with self._lock:
            log = self._log
            self._log = None
            error = self._last_error
        if log is None:
            return
        log.write(
            {
                "type": "end",
                "t": time.time(),
                "error": error,
                "cancellation_error": cancellation_error,
            }
        )
        log.close()
        logger.info("Rollout log closed", path=str(log.path))

    @staticmethod
    def _reset_policy(loaded_policy: _LoadedPolicy) -> None:
        _reset(loaded_policy.policy)
        _reset(loaded_policy.preprocessor)
        _reset(loaded_policy.postprocessor)

    def _stop_policy(self) -> bool:
        with self._lock:
            thread = self._thread
            self._stop_event.set()
            self._observation_changed.notify_all()
        if thread is not None and thread is not current_thread():
            thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        return thread is None or not thread.is_alive()

    def _cancel_after_stop_timeout(self) -> None:
        timeout_error = f"policy rollout did not stop within {DEFAULT_THREAD_JOIN_TIMEOUT} seconds"
        cancellation_error = self._cancel_trajectory()
        with self._lock:
            self._last_error = (
                f"{timeout_error}; {cancellation_error}"
                if cancellation_error is not None
                else timeout_error
            )

    def _gripper_index(self) -> int | None:
        if self.config.gripper_joint is None:
            return None
        return self.config.joint_names.index(self.config.gripper_joint)

    def _arm_joint_names(self) -> list[str]:
        return [name for name in self.config.joint_names if name != self.config.gripper_joint]

    def _trajectory(
        self,
        state: NDArray[np.float32],
        actions: NDArray[np.float32],
    ) -> JointTrajectory:
        joint_names = self._arm_joint_names()
        zeros = [0.0] * len(joint_names)
        points = [
            TrajectoryPoint(
                positions=[float(value) for value in state],
                velocities=zeros,
                time_from_start=0.0,
            )
        ]
        points.extend(
            TrajectoryPoint(
                positions=[float(value) for value in action],
                velocities=zeros,
                time_from_start=(index + 1) / self.config.fps,
            )
            for index, action in enumerate(actions)
        )
        return JointTrajectory(joint_names=joint_names, points=points)

    def _wait_for_newer_joint_state(self, previous_ts: float) -> None:
        with self._observation_changed:
            self._observation_changed.wait_for(
                lambda: self._stop_event.is_set()
                or (
                    self._latest_joint_state is not None
                    and self._latest_joint_state.ts > previous_ts
                )
            )

    def _cancel_trajectory(self) -> str | None:
        try:
            result = self._control.cancel_trajectory()
        except Exception as exc:
            logger.exception(
                "Failed to cancel policy trajectory",
            )
            return f"Failed to cancel policy trajectory: {exc}"
        if result.safe:
            return None
        message = result.message or "Policy trajectory cancellation was uncertain"
        logger.error(
            "Policy trajectory cancellation was uncertain",
            error=message,
        )
        return message


def _checkpoint_action_bounds(
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    expected_width: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    lower_tensor: torch.Tensor | None = None
    upper_tensor: torch.Tensor | None = None
    for step in postprocessor.steps:
        state = step.state_dict()
        if "action.min" in state and "action.max" in state:
            lower_tensor = state["action.min"]
            upper_tensor = state["action.max"]
            break
    if lower_tensor is None or upper_tensor is None:
        raise ValueError("Policy postprocessor has no action min/max statistics")

    lower = np.asarray(lower_tensor.detach().cpu().numpy(), dtype=np.float32)
    upper = np.asarray(upper_tensor.detach().cpu().numpy(), dtype=np.float32)
    expected_shape = (expected_width,)
    if lower.shape != expected_shape or upper.shape != expected_shape:
        raise ValueError(
            "Policy action range shape does not match configured joints: "
            f"min={lower.shape}, max={upper.shape}, expected={expected_shape}"
        )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Policy action range contains non-finite values")
    if np.any(lower > upper):
        raise ValueError("Policy action range has min greater than max")
    return lower, upper


def _reset(instance: object) -> None:
    reset = getattr(instance, "reset", None)
    if not callable(reset):
        raise TypeError(f"{type(instance).__name__} does not provide reset()")
    reset()


def _optional_int_attribute(instance: object, name: str) -> int | None:
    value = getattr(instance, name, None)
    if value is not None and not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _positive_int_attribute(instance: object, name: str) -> int:
    value = _optional_int_attribute(instance, name)
    if value is None or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value
