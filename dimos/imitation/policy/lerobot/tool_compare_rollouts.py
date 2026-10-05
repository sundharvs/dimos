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

"""Score policy rollout logs by smoothness and progress, grouped by execution mode.

Every rollout log header carries the module's ``label`` (the blueprint writes
``XARM_GRASP_POLICY_MODE`` there), so logs from a smoothing study can be laid
side by side. Per rollout: duration, measured joint-space speed, how often it
spikes, how jerky it is, how much of the path was net progress, and when the
gripper first closed. Success is yours to tally; the per-rollout rows carry the
log's timestamp so you can match them to your notes.

    python -m dimos.imitation.policy.lerobot.tool_compare_rollouts              # every log in the default dir
    python -m dimos.imitation.policy.lerobot.tool_compare_rollouts --since 19:30 # today's logs from 19:30 on
    python -m dimos.imitation.policy.lerobot.tool_compare_rollouts LOG1 LOG2 ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np

from dimos.imitation.policy.lerobot.tool_plot_rollout import (
    DEFAULT_LOG_DIR,
    STATIC_SPEED_RAD_S,
    Rollout,
    load_rollout,
    measured_speed,
    sent_origin,
)

# A measured joint-space speed above this is a lunge rather than a demo-like move.
SPIKE_SPEED_RAD_S = 0.5


@dataclass
class RolloutScore:
    path: Path
    label: str
    steps: int
    duration_s: float
    speed_mean: float
    static_frac: float
    spikes_per_min: float
    jerk_rms: float
    progress_ratio: float
    close_s: float | None
    clipped_chunks: int
    error: str | None

    def row(self) -> dict[str, Any]:
        return {
            "log": self.path.stem.removeprefix("rollout_"),
            "label": self.label,
            "steps": self.steps,
            "dur_s": round(self.duration_s, 1),
            "speed": round(self.speed_mean, 3),
            "static": round(self.static_frac, 2),
            "spikes/min": round(self.spikes_per_min, 1),
            "jerk": round(self.jerk_rms, 2),
            "progress": round(self.progress_ratio, 2),
            "close_s": None if self.close_s is None else round(self.close_s, 1),
            "clipped": self.clipped_chunks,
            "error": self.error or "",
        }


def score_rollout(rollout: Rollout, path: Path = Path("")) -> RolloutScore:
    chunks = rollout.chunks
    label = str(rollout.header.get("label") or "")
    if not label:
        replan = rollout.header.get("replan_steps", "?")
        coeff = rollout.header.get("temporal_ensemble_coeff", "?")
        label = f"replan={replan},coeff={coeff}"
    t, speed = measured_speed(rollout)
    arm = rollout.state_q[:, rollout.arm_indices] if len(rollout.state_q) else np.zeros((0, 0))
    if len(t) > 1:
        dt = np.diff(t)
        jerk = np.diff(speed) / np.where(dt > 0, dt, np.nan)
        jerk_rms = float(np.sqrt(np.nanmean(jerk**2))) if np.isfinite(jerk).any() else 0.0
        above = speed > SPIKE_SPEED_RAD_S
        spikes = int(np.sum(above[1:] & ~above[:-1]))
        minutes = max(t[-1] - t[0], 1e-9) / 60.0
        path_len = float(np.linalg.norm(np.diff(arm, axis=0), axis=1).sum())
        net = float(np.linalg.norm(arm[-1] - arm[0]))
    else:
        jerk_rms, spikes, minutes, path_len, net = 0.0, 0, 1e-9, 0.0, 0.0
    start = sent_origin(chunks[0]) if chunks else (float(t[0]) if len(t) else 0.0)
    end_t = rollout.end["t"] if rollout.end else (float(t[-1]) if len(t) else start)
    close = next(
        (sent_origin(c) - start for c in chunks if float(c.get("gripper_opening", 1.0)) < 0.5),
        None,
    )
    return RolloutScore(
        path=path,
        label=label,
        steps=len(chunks),
        duration_s=float(end_t - start),
        speed_mean=float(speed.mean()) if len(speed) else 0.0,
        static_frac=float((speed < STATIC_SPEED_RAD_S).mean()) if len(speed) else 0.0,
        spikes_per_min=spikes / minutes,
        jerk_rms=jerk_rms,
        progress_ratio=net / path_len if path_len > 0 else 0.0,
        close_s=close,
        clipped_chunks=sum(1 for c in chunks if any(c.get("clipped", []))),
        error=(rollout.end or {}).get("error"),
    )


def group_scores(scores: list[RolloutScore]) -> list[dict[str, Any]]:
    """Mean of every metric per label, plus how many rollouts closed the gripper."""
    groups: dict[str, list[RolloutScore]] = {}
    for s in scores:
        groups.setdefault(s.label, []).append(s)
    rows = []
    for label, members in groups.items():
        closes = [m.close_s for m in members if m.close_s is not None]
        rows.append(
            {
                "label": label,
                "n": len(members),
                "dur_s": round(float(np.mean([m.duration_s for m in members])), 1),
                "speed": round(float(np.mean([m.speed_mean for m in members])), 3),
                "static": round(float(np.mean([m.static_frac for m in members])), 2),
                "spikes/min": round(float(np.mean([m.spikes_per_min for m in members])), 1),
                "jerk": round(float(np.mean([m.jerk_rms for m in members])), 2),
                "progress": round(float(np.mean([m.progress_ratio for m in members])), 2),
                "closed": f"{len(closes)}/{len(members)}",
                "close_s": round(float(np.mean(closes)), 1) if closes else None,
                "errors": sum(1 for m in members if m.error),
            }
        )
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rollouts)"
    keys = list(rows[0])
    cells = [[("" if r[k] is None else str(r[k])) for k in keys] for r in rows]
    widths = [max(len(k), *(len(c[i]) for c in cells)) for i, k in enumerate(keys)]
    line = "  ".join(k.ljust(w) for k, w in zip(keys, widths, strict=True))
    body = ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join([line, *body])


def select_logs(
    paths: list[Path], directory: Path = DEFAULT_LOG_DIR, since: str | None = None
) -> list[Path]:
    if paths:
        return paths
    logs = sorted(directory.glob("rollout_*.jsonl"))
    if since:
        today = datetime.now().strftime("%Y%m%d")
        cutoff = f"rollout_{today}_{since.replace(':', '')}"
        logs = [p for p in logs if p.stem[: len(cutoff)] >= cutoff]
    return logs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="*", type=Path, help="rollout JSONL files (default: the log dir)")
    ap.add_argument("--dir", type=Path, default=DEFAULT_LOG_DIR, help="log directory")
    ap.add_argument("--since", help="only today's logs from this HH:MM on")
    ap.add_argument("--min-steps", type=int, default=5, help="skip rollouts shorter than this")
    a = ap.parse_args(argv)
    scores = []
    for path in select_logs(a.logs, a.dir, a.since):
        try:
            rollout = load_rollout(path)
        except (ValueError, KeyError, OSError) as exc:
            print(f"skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if len(rollout.chunks) < a.min_steps:
            continue
        scores.append(score_rollout(rollout, path))
    print(format_table([s.row() for s in scores]))
    print()
    print(format_table(group_scores(scores)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
