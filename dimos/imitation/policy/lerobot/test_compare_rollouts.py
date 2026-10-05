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

from dimos.imitation.policy.lerobot.tool_compare_rollouts import (
    format_table,
    group_scores,
    main,
    score_rollout,
)
from dimos.imitation.policy.lerobot.tool_plot_rollout import load_rollout

JOINTS = ["joint1", "joint2", "arm/gripper"]


def _write_log(path: Path, *, label: str, lunge: bool, close_at_chunk: int | None) -> None:
    """Two joints moving 0.3 rad over 4 one-second chunks at 15 fps, optionally with a lunge per chunk."""
    fps, n_exec = 15.0, 15
    rows: list[dict] = [
        {
            "type": "header",
            "t": 100.0,
            "joint_names": JOINTS,
            "gripper_joint": "arm/gripper",
            "fps": fps,
            "chunk_size": 45,
            "n_action_steps": n_exec,
            "replan_steps": None,
            "temporal_ensemble_coeff": None,
            "label": label,
            "policy_path": "ckpt/pretrained_model",
            "task": "t",
        }
    ]
    q = np.array([0.0, 0.0, 800.0])
    t = 100.0
    for k in range(4):
        opening = 0.0 if close_at_chunk is not None and k >= close_at_chunk else 1.0
        chunk = []
        qq = q.copy()
        for i in range(n_exec):
            step = 0.04 if (lunge and i == 0) else 0.005
            qq = qq + np.array([step, step, 0.0])
            chunk.append([float(qq[0]), float(qq[1]), opening])
        rows.append(
            {
                "type": "chunk",
                "t": t,
                "step": k * n_exec,
                "state_ts": t,
                "state": q.tolist(),
                "chunk": chunk,
                "sent": chunk,
                "executed_steps": n_exec,
                "clipped": [False] * 3,
                "infer_s": 0.01,
                "gripper_opening": opening,
                "result": "ACCEPTED",
                "sent_t": t,
            }
        )
        for i in range(n_exec):
            for sub in range(3):
                tt = t + (i + sub / 3) / fps
                prev = q if i == 0 else np.array(chunk[i - 1])
                pos = prev + (np.array(chunk[i]) - prev) * (sub / 3)
                rows.append({"type": "joint_state", "t": tt, "pos": [pos[0], pos[1], 800.0]})
        q = np.array(chunk[-1])
        t += n_exec / fps
    rows.append({"type": "end", "t": t, "error": None, "cancellation_error": None})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_lunging_rollout_scores_more_spikes_and_jerk(tmp_path: Path) -> None:
    smooth, lunging = tmp_path / "rollout_a.jsonl", tmp_path / "rollout_b.jsonl"
    _write_log(smooth, label="smooth", lunge=False, close_at_chunk=2)
    _write_log(lunging, label="lunge", lunge=True, close_at_chunk=None)
    s_smooth = score_rollout(load_rollout(smooth), smooth)
    s_lunge = score_rollout(load_rollout(lunging), lunging)
    assert s_smooth.label == "smooth" and s_lunge.label == "lunge"
    assert s_lunge.spikes_per_min > s_smooth.spikes_per_min
    assert s_lunge.jerk_rms > s_smooth.jerk_rms
    assert s_smooth.close_s == 2.0 and s_lunge.close_s is None
    assert s_smooth.progress_ratio > 0.99  # straight line, no dithering


def test_groups_average_per_label_and_count_closes(tmp_path: Path) -> None:
    for i, close in enumerate((1, None, 3)):
        _write_log(tmp_path / f"rollout_{i}.jsonl", label="m", lunge=False, close_at_chunk=close)
    scores = [score_rollout(load_rollout(p), p) for p in sorted(tmp_path.glob("*.jsonl"))]
    (row,) = group_scores(scores)
    assert row["label"] == "m" and row["n"] == 3 and row["closed"] == "2/3"
    assert row["close_s"] == 2.0
    assert "closed" in format_table([row])


def test_main_prints_both_tables(tmp_path: Path, capsys) -> None:
    _write_log(tmp_path / "rollout_x.jsonl", label="x", lunge=False, close_at_chunk=None)
    assert main(["--dir", str(tmp_path), "--min-steps", "1"]) == 0
    out = capsys.readouterr().out
    assert "label" in out and "x" in out and "closed" in out
