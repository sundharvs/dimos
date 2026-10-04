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

"""Send the arm to its home preset from the teleop keyboard."""

from __future__ import annotations

import threading
from typing import Any, Protocol

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.imitation.collection.episode_monitor import KeyPress
from dimos.spec.utils import Spec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class HomeSpec(Spec, Protocol):
    """Whatever module can move the arm home: ``ManipulationSkills`` in the arm stacks."""

    def go_home(self) -> Any: ...


class KeyboardHomeModuleConfig(ModuleConfig):
    # pygame key name, as KeyboardTeleopModule publishes it.
    home_key: str = "z"


class KeyboardHomeModule(Module):
    """Call ``go_home()`` on the home provider when the home key is pressed.

    Planning and executing the motion block for seconds, so the call runs on a
    worker thread; presses while one is in flight are ignored. The trajectory
    task outranks the keyboard's twist task, so the arm follows the planned
    motion and is the keyboard's again once it arrives.
    """

    config: KeyboardHomeModuleConfig

    keyboard: In[KeyPress]

    _home: HomeSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.keyboard.subscribe(self._on_key)))

    @rpc
    def stop(self) -> None:
        super().stop()

    @rpc
    def send_home(self) -> bool:
        """Start homing unless one is already in flight; True when started."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.info("Ignoring home request, homing already in flight")
                return False
            self._thread = threading.Thread(target=self._go_home, name="keyboard-home", daemon=True)
            self._thread.start()
            return True

    def _on_key(self, press: KeyPress) -> None:
        if press.key == self.config.home_key:
            self.send_home()

    def _go_home(self) -> None:
        logger.info("Homing from the keyboard")
        try:
            result = self._home.go_home()
        except Exception as exc:
            logger.error("Homing failed", error=str(exc))
            return
        if getattr(result, "success", True):
            logger.info("Homing finished", result=str(result))
        else:
            logger.error("Homing failed", result=str(result))


keyboard_home = KeyboardHomeModule.blueprint
