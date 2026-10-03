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

"""Save one image from the running dimOS stack, to look at what a camera sees.

    python .agents/skills/piper-hardware/scripts/grab_frame.py /tmp/frame.png
    python .agents/skills/piper-hardware/scripts/grab_frame.py /tmp/frame.png --topic /color_image

A native camera module publishes straight onto the bus, so ``Dimos.peek_stream``
never sees its frames; this subscribes to the topic itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import queue

import cv2

from dimos.core.transport_factory import make_transport
from dimos.msgs.sensor_msgs.Image import Image

FRAME_TIMEOUT_S = 8.0


class NoFrameError(RuntimeError):
    """Nothing was published on the topic in time."""


def save_frame(output: Path, topic: str = "/color_image") -> Image:
    """Write the next image published on ``topic`` to ``output``."""
    frames: queue.Queue[Image] = queue.Queue(maxsize=1)

    def keep_first(image: Image) -> None:
        if frames.empty():
            frames.put_nowait(image)

    transport = make_transport(topic, Image)
    unsubscribe = transport.subscribe(keep_first)
    try:
        image = frames.get(timeout=FRAME_TIMEOUT_S)
    except queue.Empty:
        raise NoFrameError(f"no frame on {topic} within {FRAME_TIMEOUT_S:.0f}s") from None
    finally:
        unsubscribe()
        transport.stop()
    cv2.imwrite(str(output), image.to_opencv())
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="image file to write, e.g. frame.png")
    parser.add_argument("--topic", default="/color_image")
    args = parser.parse_args()

    try:
        image = save_frame(args.output, args.topic)
    except NoFrameError as error:
        raise SystemExit(str(error)) from None
    print(f"{args.output}: {image.width}x{image.height} from {image.frame_id}")


if __name__ == "__main__":
    main()
