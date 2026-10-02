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

"""XR headset (xr-robot-teleop-server) teleop blueprints.

Pass `--simulation` to run inside MuJoCo, omit for real hardware.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.manipulators.xarm.blueprints.teleop import coordinator_teleop_xarm7
from dimos.teleop.xr_server.module import XrServerTeleopModule

# XArm7: right wrist flies the TCP, left hand signs commands and drives the gripper.
# The operator stands facing the robot's -Y. 100 Hz matches the coordinator tick.
teleop_xr_server_xarm7 = autoconnect(
    XrServerTeleopModule.blueprint(body_yaw_deg=-90.0, control_loop_hz=100.0),
    coordinator_teleop_xarm7,
).remappings(
    [
        (XrServerTeleopModule, "right_controller_output", "right_cartesian_command"),
        (XrServerTeleopModule, "right_gripper_command", "right_gripper_command"),
    ]
)


# Receiver only, no robot: bring-up rung 1. Run with `DIMOS_LOG_LEVEL=DEBUG`
# (or watch `dimos log -f`) to see left-hand signs as they are classified.
demo_xr_server_receiver = autoconnect(XrServerTeleopModule.blueprint())
