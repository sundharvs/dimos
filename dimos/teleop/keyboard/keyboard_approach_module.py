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

"""Find a prompted object and move near it from the teleop keyboard."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.imitation.collection.episode_monitor import KeyPress
from dimos.spec.utils import Spec
from dimos.teleop.keyboard.single_flight import SingleFlight


class ApproachSpec(Spec, Protocol):
    """Whatever module scans and moves near: ``PickAndPlaceModule`` in the arm stacks."""

    def find_and_move_near(self, prompts: list[str], distance: float = 0.01) -> Any: ...


class KeyboardApproachModuleConfig(ModuleConfig):
    # pygame key name, as KeyboardTeleopModule publishes it.
    approach_key: str = "n"
    # Object labels to scan for; the first detection is approached.
    prompts: list[str] = Field(min_length=1)
    # Standoff from the grasp point along its approach axis, in meters.
    distance: float = Field(default=0.05, gt=0.0)


class KeyboardApproachModule(Module):
    """Call ``find_and_move_near(prompts, distance)`` when the approach key is pressed.

    Scanning, planning and executing block for seconds, so the call runs on a
    worker thread; presses while one is in flight are ignored. The trajectory
    task outranks the keyboard's twist task, so the arm follows the planned
    motion and is the keyboard's again once it arrives.
    """

    config: KeyboardApproachModuleConfig

    keyboard: In[KeyPress]

    _approach: ApproachSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._flight = SingleFlight("keyboard-approach")

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.keyboard.subscribe(self._on_key)))

    @rpc
    def stop(self) -> None:
        super().stop()

    @rpc
    def approach(self) -> bool:
        """Start the configured scan-and-approach unless one is in flight; True when started."""
        return self._flight.start(
            lambda: self._approach.find_and_move_near(
                list(self.config.prompts), distance=self.config.distance
            )
        )

    def _on_key(self, press: KeyPress) -> None:
        if press.key == self.config.approach_key:
            self.approach()


keyboard_approach = KeyboardApproachModule.blueprint
