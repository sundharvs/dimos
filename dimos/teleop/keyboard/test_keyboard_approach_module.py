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

from __future__ import annotations

from threading import Event

import pytest
import pytest_mock

from dimos.imitation.collection.episode_monitor import KeyPress
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.spec.utils import spec_annotation_compliance
from dimos.teleop.keyboard.keyboard_approach_module import ApproachSpec, KeyboardApproachModule
from dimos.utils.testing.waiting import wait_until


def test_pick_and_place_provides_the_approach_spec() -> None:
    assert spec_annotation_compliance(PickAndPlaceModule, ApproachSpec)


def test_prompts_are_required() -> None:
    with pytest.raises(ValueError):
        KeyboardApproachModule(prompts=[])


def test_approach_key_scans_and_moves_near_once_while_in_flight(
    mocker: pytest_mock.MockerFixture,
) -> None:
    module = KeyboardApproachModule(prompts=["handle of gray bag", "bag handle"], distance=0.05)
    mocker.patch.object(module, "_approach", mocker.MagicMock(), create=True)
    started, release = Event(), Event()

    def find_and_move_near(prompts: list[str], distance: float) -> object:
        started.set()
        assert release.wait(timeout=1.0)
        return object()

    module._approach.find_and_move_near.side_effect = find_and_move_near
    try:
        module._on_key(KeyPress(key="z", ts=0.0))
        module._approach.find_and_move_near.assert_not_called()

        module._on_key(KeyPress(key="n", ts=0.0))
        assert started.wait(timeout=1.0)
        assert module.approach() is False  # already approaching
        release.set()
        wait_until(lambda: not module._flight.busy, timeout=1.0)
        module._approach.find_and_move_near.assert_called_once_with(
            ["handle of gray bag", "bag handle"], distance=0.05
        )
    finally:
        module.stop()
