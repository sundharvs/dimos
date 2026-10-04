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

import json
from pathlib import Path

import numpy as np

from dimos.imitation.policy.lerobot.tool_plot_rollout import (
    format_summary,
    load_rollout,
    plot_rollout,
    summarize,
)

JOINTS = ["joint1", "joint2", "arm/gripper"]


def _write_log(path: Path, *, stutter: bool, sent: bool = False) -> None:
    fps, n_exec, chunk_size = 15.0, 3, 6
    rows = [
        {
            "type": "header",
            "t": 100.0,
            "joint_names": JOINTS,
            "gripper_joint": "arm/gripper",
            "fps": fps,
            "chunk_size": chunk_size,
            "n_action_steps": n_exec,
            "policy_path": "ckpt/pretrained_model",
            "task": "t",
        }
    ]
    q = np.array([0.0, 0.0, 800.0])
    t = 100.0
    for _chunk_index in range(4):
        chunk = []
        qq = q.copy()
        for i in range(chunk_size):
            step = 0.0 if (stutter and i % 2) else 0.05
            qq = qq + np.array([step, step, 0.0])
            chunk.append([float(qq[0]), float(qq[1]), 1.0])
        record = {
            "type": "chunk",
            "t": t,
            "state_ts": t,
            "state": q.tolist(),
            "chunk": chunk,
            "executed_steps": n_exec,
            "clipped": [False] * 3,
            "infer_s": 0.02,
            "result": "ACCEPTED",
            "sent_t": t,
        }
        if sent:
            # The ensembled targets the runtime sent: half-steps, so they differ from the chunk.
            record["sent"] = [[(q[0] + c[0]) / 2, (q[1] + c[1]) / 2, c[2]] for c in chunk[:n_exec]]
            chunk = [list(row) for row in record["sent"]] + chunk[n_exec:]
        rows.append(record)
        for i in range(n_exec):
            for sub in range(5):  # 75 Hz joint states following the executed actions
                tt = t + (i + sub / 5) / fps
                frac = sub / 5
                prev = q if i == 0 else np.array(chunk[i - 1])
                pos = prev + (np.array(chunk[i]) - prev) * frac
                rows.append(
                    {"type": "joint_state", "t": tt, "pos": [float(pos[0]), float(pos[1]), 800.0]}
                )
        q = np.array(chunk[n_exec - 1])
        t += n_exec / fps
    rows.append({"type": "end", "t": t, "error": None, "cancellation_error": None})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_summary_separates_commanded_pauses_from_smooth_motion(tmp_path: Path) -> None:
    smooth, stutter = tmp_path / "smooth.jsonl", tmp_path / "stutter.jsonl"
    _write_log(smooth, stutter=False)
    _write_log(stutter, stutter=True)
    s_smooth = summarize(load_rollout(smooth))
    s_stutter = summarize(load_rollout(stutter))
    assert s_smooth["chunks"] == s_stutter["chunks"] == 4
    assert s_smooth["commanded_static_frac"] == 0.0
    assert s_stutter["commanded_static_frac"] > 0.3
    assert s_smooth["boundary_error_mean"] < 1e-9  # the arm followed the executed actions exactly
    assert "chunks 4" in format_summary(s_smooth)


def test_plot_writes_a_png(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    log = tmp_path / "r.jsonl"
    _write_log(log, stutter=True)
    fig = plot_rollout(load_rollout(log), joints=["joint1"], horizon=4, title="t")
    out = tmp_path / "r.png"
    fig.savefig(out)
    assert out.stat().st_size > 1000


def test_summary_and_plot_use_the_sent_targets_when_logged(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    raw, ensembled = tmp_path / "raw.jsonl", tmp_path / "sent.jsonl"
    _write_log(raw, stutter=False)
    _write_log(ensembled, stutter=False, sent=True)
    s_raw = summarize(load_rollout(raw))
    s_sent = summarize(load_rollout(ensembled))
    assert s_sent["chunks"] == 4
    # The sent half-steps start closer to the state and move slower than the raw chunk.
    assert s_sent["first_step_jump_mean"] < s_raw["first_step_jump_mean"]
    assert s_sent["commanded_speed_mean"] < s_raw["commanded_speed_mean"]
    fig = plot_rollout(load_rollout(ensembled), joints=["joint1"], title="t")
    fig.savefig(tmp_path / "sent.png")
    assert (tmp_path / "sent.png").stat().st_size > 1000
