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

from collections.abc import Iterator
from threading import Event

import pytest
import pytest_mock

from dimos.imitation.collection.episode_monitor import KeyPress
from dimos.manipulation.manipulation_skills import ManipulationSkills
from dimos.spec.utils import spec_annotation_compliance
from dimos.teleop.keyboard.keyboard_home_module import HomeSpec, KeyboardHomeModule
from dimos.utils.testing.waiting import wait_until


@pytest.fixture
def module(mocker: pytest_mock.MockerFixture) -> Iterator[KeyboardHomeModule]:
    module = KeyboardHomeModule()
    mocker.patch.object(module, "_home", mocker.MagicMock(), create=True)
    try:
        yield module
    finally:
        module.stop()


def test_manipulation_skills_provides_the_home_spec() -> None:
    assert spec_annotation_compliance(ManipulationSkills, HomeSpec)


def test_home_key_calls_go_home_once_while_in_flight(module: KeyboardHomeModule) -> None:
    started, release = Event(), Event()

    def go_home() -> object:
        started.set()
        assert release.wait(timeout=1.0)
        return object()

    module._home.go_home.side_effect = go_home

    module._on_key(KeyPress(key="x", ts=0.0))
    module._home.go_home.assert_not_called()

    module._on_key(KeyPress(key="z", ts=0.0))
    assert started.wait(timeout=1.0)
    assert module.send_home() is False  # already homing
    module._on_key(KeyPress(key="z", ts=0.0))
    release.set()
    wait_until(lambda: not (module._thread and module._thread.is_alive()), timeout=1.0)
    assert module._home.go_home.call_count == 1

    assert module.send_home() is True
    wait_until(lambda: module._home.go_home.call_count == 2, timeout=1.0)


def test_home_key_is_configurable(mocker: pytest_mock.MockerFixture) -> None:
    module = KeyboardHomeModule(home_key="home")
    mocker.patch.object(module, "_home", mocker.MagicMock(), create=True)
    module._on_key(KeyPress(key="z", ts=0.0))
    module._home.go_home.assert_not_called()
    module._on_key(KeyPress(key="home", ts=0.0))
    wait_until(lambda: module._home.go_home.call_count == 1, timeout=1.0)
    module.stop()
