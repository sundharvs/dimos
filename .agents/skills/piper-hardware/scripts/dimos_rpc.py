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

"""Run a Python snippet against the running dimOS stack, without a terminal.

    python .agents/skills/piper-hardware/scripts/dimos_rpc.py <<'EOF'
    print(app.ManipulationModule.get_state())
    EOF

``dimos shell`` needs an interactive terminal; this is the same connected ``app``
for scripts and agents. The snippet comes from stdin and sees ``app``, ``np`` and
``time``. Only ``@rpc`` and ``@skill`` methods are callable.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from dimos.porcelain.dimos import Dimos

# A stack that is still loading models answers late.
CONNECT_TIMEOUT_S = 15.0


def main() -> None:
    app = Dimos.connect(timeout=CONNECT_TIMEOUT_S)
    try:
        exec(sys.stdin.read(), {"app": app, "np": np, "time": time})
    finally:
        app.stop()


if __name__ == "__main__":
    main()
