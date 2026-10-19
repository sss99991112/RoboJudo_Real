"""Recorded-source parity followed by one bounded RoboJudo physics replay.

UniLab is never imported. The source evidence contains nominal-physics records,
not a claim about policy reliability outside these seven cases or real hardware.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from box import Box

from robojudo.config.g1.g1_standing_assist_cfg import G1StandingAssistEnvCfg, g1_unilab_standing_assist_sim
from robojudo.pipeline.unilab_standing_assist_sim_pipeline import UniLabStandingAssistSimPipeline


def controls(case, frame):
    events = []
    if case == "walk" and frame == 0:
        events = [dict(type="keyboard", name="w", pressed=True)]
    elif case == "low_stand" and frame < 11:
        events = [dict(type="keyboard", name="[", pressed=True)]
    return Box(KeyboardCtrl=dict(keyboard_event=events), COMMANDS=[])


def make_pipeline(supine=False):
    return UniLabStandingAssistSimPipeline(
        g1_unilab_standing_assist_sim(env=G1StandingAssistEnvCfg(headless=True, reset_supine=supine), ctrl=[])
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True, help="Read code only, never import it")
    p.add_argument("--source-evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_file = args.source_root / "src/unilab/algos/torch/distill/standing_assist.py"
    target_file = Path(__file__).resolve().parents[1] / "robojudo/policy/standing_assist_controller.py"

    def definitions(path):
        return [
            ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef)
        ]

    assert definitions(source_file) == definitions(target_file), "controller definitions drifted"
    cases = [
        ("stand", 0, "regressions/stand_seed0"),
        ("low_stand", 0, "regressions/low_stand_seed0_matched"),
        ("walk", 0, "regressions/walk_seed0"),
        ("supine", 0, "regressions/supine_seed0"),
        ("supine", 1, "regressions/supine_seed1"),
        ("supine", 2, "final_on_seed2"),
        ("push", 0, "regressions/push_seed0"),
    ]
    results = []
    for case, seed, relative in cases:
        d = np.load(args.source_evidence / relative / "trajectory.npz")
        pipe = make_pipeline()
        frames = len(d["decision_time"])
        try:
            base = pipe.env.get_data()
            pipe.policy.reset()
            for frame in range(frames):
                data = Box(base)
                data.recovery_control_frame = frame
                data.recovery_height = d["decision_height"][frame]
                data.policy_gravity = -d["decision_upvector"][frame]
                data.policy_gyro = d["decision_gyro"][frame]
                data.recovery_linvel = d["decision_linvel"][frame]
                data.recovery_feet_contact = d["decision_feet"][frame]
                data.dof_pos = d["decision_q"][frame]
                data.dof_vel = d["decision_dq"][frame]
                obs, extra = pipe.policy.get_observation(data, controls(case, frame))
                target = pipe.policy.get_pd_target(obs)
                np.testing.assert_array_equal(obs, d["decision_obs"][frame])
                np.testing.assert_array_equal(target, d["decision_target"][frame])
                for key in (
                    "action",
                    "proposed_target",
                    "alpha",
                    "active",
                    "stable_count",
                    "settle",
                    "assisting",
                    "attempted",
                    "assist_age",
                    "load_rejected",
                    "event",
                ):
                    np.testing.assert_array_equal(
                        extra[key], d[f"decision_{key}"][frame], err_msg=f"{case}:{frame}:{key}"
                    )
                before = pipe.policy.core.last_frame
                again, _ = pipe.policy.get_observation(data, Box(KeyboardCtrl=dict(keyboard_event=[]), COMMANDS=[]))
                np.testing.assert_array_equal(pipe.policy.get_pd_target(again), target)
                assert pipe.policy.core.last_frame == before
            results.append(
                dict(case=case, seed=seed, frames=frames, obs_action_target_route_max_error=0, duplicate_reads=frames)
            )
        finally:
            pipe.env.shutdown()
    # Exercise the real target pipeline, source's exact initial physical state,
    # and a first-class hardware-free environment. No source Python import.
    d = np.load(args.source_evidence / "final_on_seed2/trajectory.npz")
    pipe = make_pipeline(True)
    try:
        env = pipe.env
        env.data.qpos[:] = d["initial_qpos"]
        env.data.qvel[:] = d["initial_qvel"]
        env.data.qacc_warmstart[:] = 0
        mujoco.mj_forward(env.model, env.data)
        env.update()
        pipe.policy.reset()
        trace = []
        for frame in range(750):
            pipe.ctrl_manager.get_ctrl_data = lambda _data, frame=frame: controls("supine", frame)
            pipe.step()
            extra = pipe.policy._cached[1]
            after = env.get_data()
            np.testing.assert_array_equal(after.dof_pos, d["state_q"][frame])
            np.testing.assert_array_equal(after.dof_vel, d["state_dq"][frame])
            np.testing.assert_array_equal(after.policy_gyro, d["state_gyro"][frame])
            np.testing.assert_array_equal(after.recovery_linvel, d["state_linvel"][frame])
            np.testing.assert_array_equal(after.recovery_feet_contact, d["state_feet"][frame])
            np.testing.assert_array_equal(-after.policy_gravity, d["state_upvector"][frame])
            stable = bool(
                after.recovery_height >= np.float32(0.65)
                and -after.policy_gravity[2] >= np.cos(np.deg2rad(10))
                and np.linalg.norm(after.recovery_linvel) <= np.float32(0.2)
                and np.linalg.norm(after.policy_gyro) <= np.float32(0.3)
                and np.all(after.recovery_feet_contact)
            )
            assert stable == bool(d["state_stable"][frame])
            np.testing.assert_array_equal(env.data.actuator_force, d["physics_tau"][frame * 3 + 2])
            np.testing.assert_array_equal(pipe.policy.last_action, d["decision_action"][frame])
            trace.append(
                [
                    frame,
                    float(after.recovery_height),
                    float(np.rad2deg(np.arccos(np.clip(-after.policy_gravity[2], -1, 1)))),
                    float(np.linalg.norm(after.recovery_linvel)),
                    float(np.linalg.norm(after.policy_gyro)),
                    int(stable),
                    extra["expert"],
                    extra["settle"],
                ]
            )
        np.testing.assert_array_equal(np.asarray(trace), d["trace"])
        np.savez_compressed(args.output / "supine_seed2.npz", trace=np.asarray(trace))
        result = dict(
            controller_definitions_equal=True,
            offline=results,
            total_offline_frames=sum(x["frames"] for x in results),
            physics_steps=750,
            physics_substeps=2250,
            physical_state_action_force_trace_exact=True,
            imported_unilab=any(name == "unilab" or name.startswith("unilab.") for name in sys.modules),
        )
        assert not result["imported_unilab"]
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        pipe.env.shutdown()


if __name__ == "__main__":
    main()
