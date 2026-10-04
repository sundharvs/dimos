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

"""Run one blocking robot action at a time from key presses."""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any

from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class SingleFlight:
    """Run a blocking callable on a worker thread, refusing a second while one runs.

    Key presses arrive on the stream thread and skills block for seconds, so
    each one runs detached; a press during a run is dropped rather than queued.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, action: Callable[[], Any]) -> bool:
        """Start ``action`` unless one is in flight; True when started."""
        with self._lock:
            if self.busy:
                logger.info("Ignoring request, action already in flight", action=self._name)
                return False
            self._thread = threading.Thread(
                target=self._run, args=(action,), name=self._name, daemon=True
            )
            self._thread.start()
            return True

    def _run(self, action: Callable[[], Any]) -> None:
        logger.info("Action started", action=self._name)
        try:
            result = action()
        except Exception as exc:
            logger.error("Action failed", action=self._name, error=str(exc))
            return
        if getattr(result, "success", True):
            logger.info("Action finished", action=self._name, result=str(result))
        else:
            logger.error("Action failed", action=self._name, result=str(result))
