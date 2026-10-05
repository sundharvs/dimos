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

"""Plot what a LeRobot policy predicted during a rollout against what the arm did.

``LeRobotPolicyModule`` writes one JSONL file per rollout (``rollout_log_dir``,
default ``~/.local/state/dimos/policy_rollouts``): a header, every coordinator
joint state while the rollout ran, and one ``chunk`` record per inference with
the full predicted action chunk and, when the runtime ensembles, the targets it
actually sent. This tool renders, per joint, the measured position (black) with
every submission overlaid from the moment it was sent: the steps executed before
the next submission solid, the rest of the sent trajectory dashed, and the raw
predicted chunk dotted where it differs from what was sent.
The bottom panel compares the joint-space speed the policy commanded with the
speed the arm measured, which is where stutter shows up: pauses inside the
commanded chunks mean the policy predicts stop-and-go, pauses only in the
measured speed mean execution (chunk boundaries, velocity limits) adds them.

    python -m dimos.imitation.policy.lerobot.tool_plot_rollout            # latest log -> PNG next to it
    python -m dimos.imitation.policy.lerobot.tool_plot_rollout LOG --show  # also open a window
    python -m dimos.imitation.policy.lerobot.tool_plot_rollout LOG --t0 5 --t1 20 --joints joint1 joint3

The summary printed alongside quantifies the same thing per chunk.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import pairwise
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dimos.constants import STATE_DIR

DEFAULT_LOG_DIR = STATE_DIR / "policy_rollouts"
STATIC_SPEED_RAD_S = 0.02


@dataclass
class Rollout:
    header: dict[str, Any]
    joint_names: list[str]
    fps: float
    state_t: NDArray[np.float64]  # (N,)
    state_q: NDArray[np.float64]  # (N, J)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    end: dict[str, Any] | None = None

    @property
    def t0(self) -> float:
        starts = [float(self.state_t[0])] if len(self.state_t) else []
        starts += [float(c["t"]) for c in self.chunks]
        return min(starts) if starts else 0.0

    @property
    def arm_indices(self) -> list[int]:
        gripper = self.header.get("gripper_joint")
        return [i for i, name in enumerate(self.joint_names) if name != gripper]


def load_rollout(path: Path) -> Rollout:
    header: dict[str, Any] | None = None
    times: list[float] = []
    positions: list[list[float]] = []
    chunks: list[dict[str, Any]] = []
    end: dict[str, Any] | None = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("type")
            if kind == "header":
                header = rec
            elif kind == "joint_state":
                times.append(float(rec["t"]))
                positions.append([float(v) for v in rec["pos"]])
            elif kind == "chunk":
                chunks.append(rec)
            elif kind == "end":
                end = rec
    if header is None:
        raise ValueError(f"{path}: no header record")
    joint_names = list(header["joint_names"])
    order = np.argsort(times) if times else np.array([], dtype=int)
    state_t = np.asarray(times, dtype=np.float64)[order]
    state_q = (
        np.asarray(positions, dtype=np.float64)[order]
        if positions
        else np.zeros((0, len(joint_names)))
    )
    return Rollout(
        header=header,
        joint_names=joint_names,
        fps=float(header.get("fps") or 15.0),
        state_t=state_t,
        state_q=state_q,
        chunks=chunks,
        end=end,
    )


def measured_speed(
    rollout: Rollout, window_s: float | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Joint-space speed (rad/s) of the measured arm joints, differenced over one policy step."""
    arm = rollout.arm_indices
    if len(rollout.state_t) < 2:
        return np.zeros(0), np.zeros(0)
    window = window_s if window_s is not None else 1.0 / rollout.fps
    t = rollout.state_t
    q = rollout.state_q[:, arm]
    speeds = np.zeros(len(t))
    j = 0
    for i in range(len(t)):
        while t[i] - t[j] > window and j < i:
            j += 1
        dt = t[i] - t[j]
        speeds[i] = np.linalg.norm(q[i] - q[j]) / dt if dt > 0 else 0.0
    return t, speeds


def sent_actions(chunk: dict[str, Any]) -> NDArray[np.float64]:
    """The targets the runtime submitted for this record: ensembled if logged, else the raw chunk."""
    return np.asarray(chunk.get("sent", chunk["chunk"]), dtype=np.float64)


def sent_origin(chunk: dict[str, Any]) -> float:
    """When the submitted trajectory started: its send time, else the prediction time."""
    return float(chunk.get("sent_t", chunk["t"]))


def sent_offset(chunk: dict[str, Any]) -> int:
    """Steps the first sent target was delayed by (a first-step ramp), else 0."""
    return int(chunk.get("sent_offset", 0))


def commanded_speed(rollout: Rollout, chunk: dict[str, Any]) -> NDArray[np.float64]:
    """Joint-space speed (rad/s) between consecutive sent arm targets, prefixed with the step from the state."""
    arm = rollout.arm_indices
    actions = sent_actions(chunk)[:, arm]
    state = np.asarray(chunk["state"], dtype=np.float64)[arm]
    path = np.vstack([state, actions])
    speeds: NDArray[np.float64] = np.linalg.norm(np.diff(path, axis=0), axis=1) * rollout.fps
    # A first-step ramp spreads the move to the first target over the skipped steps.
    speeds[0] /= sent_offset(chunk) + 1
    return speeds


def summarize(rollout: Rollout) -> dict[str, Any]:
    chunks = rollout.chunks
    if not chunks:
        return {"chunks": 0}
    executed = [commanded_speed(rollout, c)[: int(c["executed_steps"])] for c in chunks]
    full = [commanded_speed(rollout, c) for c in chunks]
    exec_cat = np.concatenate(executed)
    first_jump = [
        float(
            np.linalg.norm(
                sent_actions(c)[0][rollout.arm_indices]
                - np.asarray(c["state"], dtype=np.float64)[rollout.arm_indices]
            )
        )
        for c in chunks
    ]
    # Tracking error at chunk boundaries: last executed action of chunk k vs the
    # state the next chunk was predicted from.
    boundary_err: list[float] = []
    for prev, nxt in pairwise(chunks):
        last = sent_actions(prev)[int(prev["executed_steps"]) - 1][rollout.arm_indices]
        nxt_state = np.asarray(nxt["state"], dtype=np.float64)[rollout.arm_indices]
        boundary_err.append(float(np.linalg.norm(last - nxt_state)))
    periods = np.diff([sent_origin(c) for c in chunks]) if len(chunks) > 1 else np.zeros(0)
    _t, meas = measured_speed(rollout)
    summary: dict[str, Any] = {
        "chunks": len(chunks),
        "duration_s": sent_origin(chunks[-1])
        - sent_origin(chunks[0])
        + int(chunks[-1]["executed_steps"]) / rollout.fps,
        "results": {
            name: sum(1 for c in chunks if c.get("result") == name)
            for name in {c.get("result") for c in chunks}
        },
        "infer_s_mean": float(np.mean([c["infer_s"] for c in chunks])),
        "infer_s_max": float(np.max([c["infer_s"] for c in chunks])),
        "chunk_period_s_mean": float(periods.mean()) if len(periods) else float("nan"),
        "commanded_speed_mean": float(exec_cat.mean()),
        "commanded_speed_max": float(exec_cat.max()),
        "commanded_static_frac": float((exec_cat < STATIC_SPEED_RAD_S).mean()),
        "commanded_static_frac_full_chunk": float(
            (np.concatenate(full) < STATIC_SPEED_RAD_S).mean()
        ),
        "measured_static_frac": float((meas < STATIC_SPEED_RAD_S).mean())
        if len(meas)
        else float("nan"),
        "measured_speed_mean": float(meas.mean()) if len(meas) else float("nan"),
        "first_step_jump_mean": float(np.mean(first_jump)),
        "first_step_jump_max": float(np.max(first_jump)),
        "boundary_error_mean": float(np.mean(boundary_err)) if boundary_err else float("nan"),
        "boundary_error_max": float(np.max(boundary_err)) if boundary_err else float("nan"),
        "clipped_chunks": sum(1 for c in chunks if any(c.get("clipped", []))),
        "error": (rollout.end or {}).get("error"),
    }
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    if summary.get("chunks", 0) == 0:
        return "no chunk records (rollout never predicted)"
    lines = [
        f"chunks {summary['chunks']} over {summary['duration_s']:.1f} s, results {summary['results']}, "
        f"period {summary['chunk_period_s_mean']:.2f} s, inference {summary['infer_s_mean'] * 1000:.0f} ms mean / {summary['infer_s_max'] * 1000:.0f} ms max",
        f"commanded speed (executed steps): mean {summary['commanded_speed_mean']:.3f} rad/s, max {summary['commanded_speed_max']:.2f}, "
        f"static {summary['commanded_static_frac']:.0%} (full chunks {summary['commanded_static_frac_full_chunk']:.0%})",
        f"measured speed: mean {summary['measured_speed_mean']:.3f} rad/s, static {summary['measured_static_frac']:.0%}",
        f"first-step jump |a0 - state|: mean {summary['first_step_jump_mean']:.3f} rad, max {summary['first_step_jump_max']:.3f}",
        f"chunk-boundary tracking error: mean {summary['boundary_error_mean']:.3f} rad, max {summary['boundary_error_max']:.3f}",
        f"chunks with clipped actions: {summary['clipped_chunks']}",
    ]
    if summary.get("error"):
        lines.append(f"rollout ended with error: {summary['error']}")
    return "\n".join(lines)


def plot_rollout(
    rollout: Rollout,
    *,
    joints: list[str] | None = None,
    t0: float | None = None,
    t1: float | None = None,
    horizon: int | None = None,
    title: str = "",
) -> Any:
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use(matplotlib.get_backend())
    import matplotlib.pyplot as plt

    names = rollout.joint_names
    picked = [i for i, n in enumerate(names) if joints is None or n in joints]
    origin = rollout.t0
    fig, axes = plt.subplots(len(picked) + 1, 1, figsize=(14, 2.0 * len(picked) + 3), sharex=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab20")
    rel_state_t = rollout.state_t - origin
    gripper = rollout.header.get("gripper_joint")
    for ax, j in zip(axes[:-1], picked, strict=False):
        if len(rollout.state_t):
            ax.plot(
                rel_state_t,
                rollout.state_q[:, j],
                color="black",
                lw=1.2,
                label="measured",
                zorder=3,
            )
        if names[j] == gripper:
            # The gripper's action is a normalized opening (0 closed .. 1 open), its state is native.
            ax.set_ylabel(f"{names[j]}\n(state, native)", fontsize=9)
            ax = ax.twinx()
            ax.set_ylabel("predicted opening\n0 closed .. 1 open", fontsize=9)
            ax.set_ylim(-0.05, 1.05)
        for k, c in enumerate(rollout.chunks):
            sent = sent_actions(c)
            n_exec = min(int(c["executed_steps"]), sent.shape[0])
            steps = sent.shape[0] if horizon is None else min(horizon, sent.shape[0])
            start = sent_origin(c) - origin
            ts = start + (np.arange(steps) + 1 + sent_offset(c)) / rollout.fps
            color = cmap(k % 20)
            ax.plot([c["t"] - origin], [c["state"][j]], marker="o", ms=3, color=color, zorder=4)
            ax.plot(
                np.r_[start, ts[:n_exec]],
                np.r_[c["state"][j], sent[:n_exec, j]],
                color=color,
                lw=1.4,
                zorder=2,
            )
            if steps > n_exec:
                ax.plot(
                    ts[n_exec - 1 :],
                    sent[n_exec - 1 : steps, j],
                    color=color,
                    lw=0.8,
                    ls="--",
                    alpha=0.5,
                    zorder=1,
                )
            if "sent" in c:
                raw = np.asarray(c["chunk"], dtype=np.float64)
                raw_steps = raw.shape[0] if horizon is None else min(horizon, raw.shape[0])
                ax.plot(
                    start + (np.arange(raw_steps) + 1) / rollout.fps,
                    raw[:raw_steps, j],
                    color=color,
                    lw=0.6,
                    ls=":",
                    alpha=0.35,
                    zorder=1,
                )
            ax.axvline(start, color="grey", lw=0.4, ls=":", zorder=0)
        if names[j] != gripper:
            ax.set_ylabel(names[j], fontsize=9)
        ax.grid(True, alpha=0.3)
    ax = axes[-1]
    mt, ms = measured_speed(rollout)
    if len(mt):
        ax.plot(mt - origin, ms, color="black", lw=1.0, label="measured (arm joints)")
    for k, c in enumerate(rollout.chunks):
        sp = commanded_speed(rollout, c)
        n_exec = min(int(c["executed_steps"]), len(sp))
        ts = sent_origin(c) - origin + (np.arange(len(sp)) + 1 + sent_offset(c)) / rollout.fps
        ax.step(
            ts[:n_exec],
            sp[:n_exec],
            where="post",
            color=cmap(k % 20),
            lw=1.2,
            label="commanded (executed)" if k == 0 else None,
        )
        if horizon is None or horizon > n_exec:
            ax.step(
                ts[n_exec - 1 :],
                sp[n_exec - 1 :],
                where="post",
                color=cmap(k % 20),
                lw=0.7,
                ls="--",
                alpha=0.5,
            )
    ax.axhline(
        STATIC_SPEED_RAD_S,
        color="red",
        lw=0.6,
        ls=":",
        label=f"static threshold {STATIC_SPEED_RAD_S} rad/s",
    )
    ax.set_ylabel("joint-space speed\n[rad/s]", fontsize=9)
    ax.set_xlabel("time since rollout start [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    if t0 is not None or t1 is not None:
        ax.set_xlim(t0, t1)
    fig.suptitle(
        title
        or f"{Path(rollout.header.get('policy_path', '')).parent.name}  {len(rollout.chunks)} chunks",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def latest_log(directory: Path = DEFAULT_LOG_DIR) -> Path:
    logs = sorted(directory.glob("rollout_*.jsonl"))
    if not logs:
        raise SystemExit(f"no rollout logs in {directory}")
    return logs[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "log", nargs="?", type=Path, help=f"rollout JSONL (default: newest in {DEFAULT_LOG_DIR})"
    )
    ap.add_argument("--out", type=Path, help="PNG path (default: next to the log)")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    ap.add_argument("--joints", nargs="+", help="subset of joint names to plot")
    ap.add_argument("--t0", type=float, help="window start [s since rollout start]")
    ap.add_argument("--t1", type=float, help="window end [s]")
    ap.add_argument(
        "--horizon", type=int, help="predicted steps to draw per chunk (default: the whole chunk)"
    )
    a = ap.parse_args(argv)
    path = a.log or latest_log()
    rollout = load_rollout(path)
    print(f"{path}\n{format_summary(summarize(rollout))}")
    if not a.show:
        import matplotlib

        matplotlib.use("Agg")
    fig = plot_rollout(
        rollout, joints=a.joints, t0=a.t0, t1=a.t1, horizon=a.horizon, title=path.stem
    )
    out = a.out or path.with_suffix(".png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    if a.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
