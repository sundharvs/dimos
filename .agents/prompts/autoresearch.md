# Autoresearch: skill discovery on a real robot

You are running an autonomous research loop on a robot in this repository. You are given a task. The work has two phases, measured separately:

1. **Solve.** Get the robot to complete the task once, verified, in as little wall time and as few tokens as you can. This is the number compared against an end-to-end baseline (a vision-language model that moves the arm one tool call at a time until it succeeds), so nothing belongs in this phase that does not bring the first success closer.
2. **Distil.** Turn what worked into validated dimos skills, so that the next run of this task, and the next task, costs almost nothing. This is where the approach pays back: the baseline pays its full cost on every run, a committed skill is one call.

The method follows ASPIRE (Lu et al., "Agentic Skills Discovery for Robotics", arXiv:2607.00272), adapted to one physical rig with a human operator nearby.

**Task:** <!-- replace this line with the task, e.g. "pick up the yellow bin" -->

**Success criterion:** <!-- what a camera frame must show for one trial to count, e.g. "the bin is held clear of the table" -->

**Phase 2 bar:** <!-- e.g. "8 of 10 held-out trials", or "skip" to stop after Phase 1 -->

If the task or the success criterion is still a placeholder, ask for it once, then proceed. A missing Phase 2 bar means 4 of 5.

## The clock

Create the run directory `~/.local/state/dimos/autoresearch/<task-slug>/` and append a line to `clock.jsonl` in it at each of these events, with `date -Is` and the event name: `solve_start` (your first action), `attempt` (each hardware attempt, with a one-line note of what was tried and what happened), `first_success`, `distil_start`, `distil_end`. Wall time and tokens per phase are computed afterwards from these marks and the session's usage log. Do not estimate your own token use.

If `clock.jsonl` already exists, you are resuming: read it and continue in the phase it shows.

## Phase 1: Solve

Spend on the task, not on preparation.

- **Read only what you need to operate the rig**: its skill under `.agents/skills/` (for the Piper arm: `.agents/skills/piper-hardware/SKILL.md`). If a stack is already running and healthy, use it; otherwise run the preflight and start one.
- **Use the library first.** List what the robot can already do: the `@skill` methods of the modules in the running blueprint (`dimos mcp list-tools` when it includes `McpServer`, or the rig skill's RPC helper). If existing skills cover the task, call them and you are nearly done. Depth, perception and the planner are the advantage over the baseline; do not hand-drive the arm pose by pose when a skill does it.
- **Record everything, read little.** Every primitive call (a perception query, a plan, a motion, a gripper command) writes a trace entry to the run directory as it happens: the call, its inputs, its return value, a frame from each camera before and after, and what the program believed (object pose, gripper width, the check it applied and the value it saw). Writing traces costs no tokens. Reading images does, so open frames when they decide something: before a contact motion that rests on a belief nothing has checked, when a result is ambiguous, and to verify success. Crop or downscale to the region that matters.
- **Verify with a frame.** The task is complete only when a frame shows the success criterion. A script that prints success, a trajectory that returns `COMPLETED`, a gripper width, a joint reading: these are evidence about the program, not about the world. One clear frame is enough. Mark `first_success` and go to Phase 2.

Leave out of this phase: reading documentation beyond the rig skill, a debug set, repeated trials, writing modules or tests, refactoring, lint.

### When an attempt fails

This is where ASPIRE's findings apply. In its ablation, giving the agent execution traces took success from 14% to 62%, and searching over several candidate programs added another 10 points.

1. **Read the trace.** Find the first primitive where the world and the program's belief came apart. Now the frames are worth their tokens.
2. **Name the failure and a hypothesis** that the trace supports and one experiment could refute ("planner returns `JOINT_LIMITS` for straight-down poses beyond radius r").
3. **Measure instead of assuming.** Reach limits, real tool height, table height and object sizes have each been wrong when assumed on this rig. Check feasibility offline (IK, reach, planning) before spending a hardware attempt.
4. **Repair with a script** that drives the running stack through its RPCs and existing skills. A script is the fast way to iterate; it becomes a module only in Phase 2.
5. **Change the idea, not the constant.** If a second adjustment of the same idea fails, list repairs that differ in kind (a different grasp, viewpoint, check, order of operations, or a fixture the operator sets up), drop those an offline check rules out, and try the best. Keep the best program so far and build on it.

Operator resets are the slowest step. Prefer programs that restore the scene themselves, and batch what you need from the operator into one request.

Stop Phase 1 without success only when you can show from the traces that the rig cannot do the task as posed. Say what would have to change.

## Phase 2: Distil

Mark `distil_start`. Skip this phase only if the Phase 2 bar says so. If Phase 1 succeeded with existing skills alone and nothing was repaired, there is nothing new to distil: go straight to "Repeat cost" below.

### Validate

One success is a demonstration. Before anything becomes a skill:

- Choose a **debug set**: a handful of configurations (object positions, orientations, instances). Record it in the run directory. Run the program on it and repair what fails, using the same diagnose-and-search steps as in Phase 1.
- Then run a **held-out set**: configurations the program was not tuned on. Record successes over trials, each verified by a frame. This number is what you report and what a skill's validation rate cites. If it falls well short of the debug rate, the program fitted the debug set: add the held-out failures to the debug set, repair, and choose a fresh held-out set.

### Write the skill

Ask what in the validated program would help on a different task. That part becomes a skill, and a skill here is a dimos skill: a `@skill` method on a `Module`, composed into a runnable blueprint. That way what you discover is callable by the next task program over RPC, by an LLM agent as a tool, and over MCP, instead of being advice someone has to re-implement.

```python
class CapSkills(Module):
    config: CapSkillsConfig          # rig-specific numbers live in config, set by the blueprint
    _manipulation: ManipulationSpec  # other modules are reached through Spec protocols

    @skill
    def unscrew_cap(self, x: float, y: float, top_z: float) -> SkillResult:
        """Unscrew the cap of an upright, clamped bottle and keep it in the gripper.

        Use when the cap top is within straight-down reach. Looks at the bottle
        after each stroke; stops if the grip is not cap width.

        Args:
            x: Cap centre in world, metres.
            y: Cap centre in world, metres.
            top_z: Height of the cap top in world, metres.
        """
```

Follow `AGENTS.md` ("Agent System", "Adding a New Skill") and the nearest existing module for the details. What ASPIRE puts in a skill entry maps onto the schema like this:

| ASPIRE field | Where it goes |
|---|---|
| When to apply, and when not | The `@skill` docstring. It is the tool description the LLM sees on every call, so keep it to what a caller needs to choose and call the skill. |
| Repair strategy | The method body, built from other modules' RPCs through `Spec` protocols. The verification that made it work (look, compare, retract when unsure) is part of the skill, not of the caller. |
| Failure signature | The return value: a `SkillResult` or string that names what was seen and what failed, so a caller can branch on it. Never return success the skill did not observe. |
| Task and rig quirks | `ModuleConfig` fields with defaults, overridden in the rig's blueprint with a comment saying how each number was measured and when. No coordinates, pixel boxes or thresholds hard-coded in the method. |
| Origin and validation rate | The module docstring: origin task, date, k of n on held-out configurations, what is untested. |
| Guidance on combining skills | The rig's system prompt, when the LLM needs it. |

Then wire it up:

- Put the module beside its peers (`dimos/manipulation/` for arm skills, or the robot's own package), with a `test_<module>_unit.py` covering its logic without hardware.
- Add it to a blueprint under the robot's `blueprints/` directory with `autoconnect(...)`, expose the blueprint as a module-level variable, and regenerate the registry: `pytest dimos/robot/test_all_blueprints_generation.py`. Never edit `all_blueprints.py` by hand.
- Run the held-out trials through the blueprint (`dimos run <blueprint>`, then call the skill over RPC or `dimos mcp call`), so what you validated is what you committed.
- `uv run mypy` and ruff must pass on what you add.

Admission rules, because a wrong skill misleads every later run:

- Admit only what passed held-out validation through its blueprint. A skill that has never completed a run on the robot is not a skill yet; keep it on the branch as work in progress and say so in its docstring.
- Separate the transferable pattern from the task's quirks, as in the table.
- Before adding, look for an existing skill or module that covers it and extend that one. When you find a skill that is wrong, stale or too specific, fix or remove it and say why in the commit.
- Prefer a few composable skills (find, grasp, verify, recover) over one that performs the whole task; the whole task is a short skill that calls them.

Knowledge that is about operating the rig as an agent, not about what the robot can do (bring-up, debugging workflows, how to grab a frame), is not a robot skill. It belongs in the rig's `.agents/skills/<rig>/SKILL.md` and its `scripts/`.

Commit as you go on the current branch. Do not push.

### Repeat cost

Finish by running the task once more from a reset scene using only committed skills: one call, one verification frame. Mark it in `clock.jsonl` as `repeat_start` and `repeat_end`. This is the cost of every future run, and the figure that the Phase 1 and Phase 2 spend buys. Then mark `distil_end`.

## Working on real hardware

ASPIRE learned in simulation, where a bad program costs nothing. Here it can break things, so these rules hold in both phases. Saving time or tokens is never a reason to skip one:

- **Never claim a state you have not seen.** Before a contact motion that depends on where something is, and before any report, there must be a frame or a semantic check (the detector, a depth measurement) behind the belief, not a proxy such as a joint reading or a `COMPLETED`. Use a view that can see the contact: a wrist camera usually cannot see its own gripper touch anything.
- **When a check is ambiguous, retract and look.** Never repeat a contact motion (press, descend, twist, re-grip) on an unverified assumption.
- **Tell the planner about obstacles** it cannot know (tall objects, fixtures), and retract upward before sweeping past them.
- **Stopping is a motion.** Park the arm in its rest posture with a planned move before stopping or restarting the stack. Prefer changing parameters over RPC to restarting while the arm is extended.
- **If the operator says stop:** kill the running program, cancel motion, send nothing further, then report where the arm is from a frame, and what you do and do not know.
- Stay inside the workspace, speeds and forces the rig's skill describes. If a candidate needs more, it is a question for the operator.

## Autonomy and the operator

Design, tooling, thresholds, retries, what to try next and commits on this branch are yours to decide. Ask the operator only for what you cannot do: physically resetting or swapping objects, privileged commands, hardware you cannot reach, a fact only they have. When you ask, say exactly what to place where, and carry on with whatever does not depend on the answer.

## Finishing

1. Park the arm and confirm from a frame.
2. If Phase 2 ran: every admitted skill, its blueprint and its test are committed, and the blueprint registry is regenerated.
3. Report, keeping the phases apart:
   - **Solve:** succeeded or not, number of hardware attempts and operator resets, and the `solve_start` and `first_success` times.
   - **Distil:** the held-out result as k of n, and the skills added or changed.
   - **Repeat:** wall time of the repeat run.
   - What was verified on the robot and what was not, the failure modes that remain, and the state the scene was left in.

Report failures as plainly as successes.
