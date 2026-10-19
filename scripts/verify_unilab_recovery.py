"""Source parity first, then bounded CPU MuJoCo episodes (never hardware).

Run using UniLab's uv interpreter, with RoboJudo on PYTHONPATH. Reports and
sampled rollout NPZ files are local evidence, not deployment approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from box import Box

from robojudo.config.g1.g1_recovery_cfg import G1UniLabRecoveryEnvCfg, g1_unilab_recovery_sim
from robojudo.pipeline.unilab_recovery_sim_pipeline import UniLabRecoverySimPipeline
from robojudo.policy.recovery_route import ROLES, ROUTING_CONTRACT, STUDENT_SHA256, RecoveryRoute


def source_environment(root):
    from omegaconf import OmegaConf
    from unilab.training import BackendAdapter, create_env, ensure_registries

    ensure_registries()
    cfg = OmegaConf.load(root / "conf/distill/config.yaml")
    del cfg["defaults"]
    cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "conf/distill/task/g1_recovery_combined/mujoco.yaml"))
    cfg.env.noise_config.level = 0.0  # deterministic I/O comparison, not a new quality protocol
    return create_env(
        cfg,
        num_envs=1,
        sim_backend="mujoco",
        task_name="G1RecoveryCombined",
        env_cfg_override=BackendAdapter(cfg, root_dir=root, algo_name="distill").build_task_env_cfg_override(),
    )


def source_update(info, data, source):
    from unilab.envs.locomotion.g1.recovery_combined import update_recovery_route

    update_recovery_route(
        info,
        height=np.array([data.recovery_height], np.float32),
        gravity=-np.asarray(data.policy_gravity, np.float32)[None],
        linvel=np.asarray(data.recovery_linvel, np.float32)[None],
        gyro=np.asarray(data.policy_gyro, np.float32)[None],
        feet=np.asarray(data.recovery_feet_contact)[None],
        steps=np.array([data.recovery_control_frame]),
        recovery=source.cfg.recovery,
        routing=source.cfg.recovery_routing,
        ctrl_dt=source.cfg.ctrl_dt,
    )
    if info["recovery_active"][0]:
        info["gait_phase"][:] = 0


def new_info(phase):
    return {
        "commands": np.zeros((1, 3), np.float32),
        "height_commands": np.full((1, 1), 0.754, np.float32),
        "gait_phase": phase.copy()[None],
        "current_actions": np.zeros((1, 29), np.float32),
    }


def check_observation_and_action(source, student, wrapper, data, ctrl, info):
    from unilab.algos.torch.distill.recovery_integration import recovery_expert_indices

    obs, extras = wrapper.get_observation(data, ctrl)
    info["commands"][:] = extras["requested_commands"]
    info["height_commands"][:] = extras["requested_height"]
    source_update(info, data, source)
    expected = source._compute_obs(
        info,
        np.asarray(data.recovery_linvel)[None],
        np.asarray(data.policy_gyro)[None],
        -np.asarray(data.policy_gravity)[None],
        np.asarray(data.dof_pos)[None],
        np.asarray(data.dof_vel)[None],
    )["obs"][0]
    error = float(np.max(np.abs(obs - expected)))
    np.testing.assert_allclose(obs, expected, atol=1e-6, rtol=0)
    index = int(recovery_expert_indices(info)[0])
    assert index == extras["expert_index"]
    assert int(info["recovery_route_stable_steps"][0]) == wrapper.route.stable_steps
    assert int(info["recovery_settle_remaining"][0]) == wrapper.route.settle_remaining
    before = (wrapper.route.stable_steps, wrapper.route.settle_remaining, wrapper.route.active)
    wrapper.get_observation(data, {"KeyboardCtrl": {"keyboard_event": []}})
    source_update(info, data, source)
    assert before == (wrapper.route.stable_steps, wrapper.route.settle_remaining, wrapper.route.active)
    pd_target = wrapper.get_pd_target(obs)
    with torch.inference_mode():
        action = student.experts[index](torch.from_numpy(expected)[None]).numpy()[0]
    action_error = float(np.max(np.abs(wrapper.last_action - action)))
    np.testing.assert_allclose(wrapper.last_action, action, rtol=0, atol=1e-6)
    # Call the real source execution owner, including its default-angle offset.
    state = type("SourceState", (), {"info": info})()
    source_target = source.apply_action(action[None], state)[0]
    target_error = float(np.max(np.abs(pd_target - source_target)))
    np.testing.assert_allclose(pd_target, source_target, atol=1e-6, rtol=0)
    return obs, extras, pd_target, (error, action_error, target_error)


def verify_offline(source, student, pipeline):
    """Synthetic same-state comparison is deliberately separate from physics."""
    from unilab.algos.torch.distill.recovery_integration import RECOVERY_ROLES, RECOVERY_ROUTING_CONTRACT

    assert RECOVERY_ROLES == ROLES and RECOVERY_ROUTING_CONTRACT == ROUTING_CONTRACT
    assert not source.cfg.reward_config.gait_constraint.enabled
    assert not source.cfg.control_config.simulate_action_latency
    from unilab.envs.locomotion.g1.recovery_combined import update_recovery_route

    boundary_cases = 0
    for height in (np.float32(0.65), np.nextafter(np.float32(0.65), np.float32(0)), np.float32(0.4)):
        for linear_speed in (np.float32(0.2), np.nextafter(np.float32(0.2), np.float32(1))):
            for angular_speed in (np.float32(0.3), np.nextafter(np.float32(0.3), np.float32(1))):
                route = RecoveryRoute(active=True, stable_steps=49, last_frame=0)
                reference = {
                    "recovery_active": np.array([True]),
                    "recovery_route_stable_steps": np.array([49]),
                    "recovery_route_last_step": np.array([0]),
                    "recovery_settle_remaining": np.array([0]),
                }
                velocity = np.array([linear_speed, 0, 0], np.float32)
                gyro = np.array([angular_speed, 0, 0], np.float32)
                route.update(frame=1, height=float(height), upvector=[0, 0, 1], linvel=velocity, gyro=gyro, feet=[1, 1])
                update_recovery_route(
                    reference,
                    height=np.array([height]),
                    gravity=np.array([[0, 0, 1]], np.float32),
                    linvel=velocity[None],
                    gyro=gyro[None],
                    feet=np.array([[True, True]]),
                    steps=np.array([1]),
                    recovery=source.cfg.recovery,
                    routing=source.cfg.recovery_routing,
                    ctrl_dt=0.02,
                )
                assert route.active == bool(reference["recovery_active"][0])
                assert route.stable_steps == int(reference["recovery_route_stable_steps"][0])
                assert route.settle_remaining == int(reference["recovery_settle_remaining"][0])
                boundary_cases += 1
    wrapper = pipeline.policy
    wrapper.reset()
    info = new_info(wrapper.gait_phase)
    base = pipeline.env.get_data()
    rng = np.random.default_rng(19)
    errors = np.zeros(3)
    routes = [0, 0, 0]
    # Fall -> broken stability -> full window -> settle -> latest stop request
    # -> walking -> refall. Repeated route reads must not consume a frame.
    for frame in range(340):
        data = Box(base)
        data.recovery_control_frame = frame
        data.dof_pos = (np.asarray(wrapper.default_pos, np.float32) + rng.normal(0, 0.1, 29)).astype(np.float32)
        data.dof_vel = rng.normal(0, 0.2, 29).astype(np.float32)
        data.policy_gyro = np.zeros(3, np.float32)
        data.policy_gravity = np.array([0, 0, -1], np.float32)
        data.recovery_height = np.float32(0.754)
        data.recovery_linvel = np.zeros(3, np.float32)
        data.recovery_feet_contact = np.ones(2, bool)
        if frame < 20 or frame == 300:
            data.recovery_height = np.float32(0.30)
            data.policy_gravity = np.array([-1, 0, 0], np.float32)
        if frame == 50:
            data.recovery_feet_contact = np.array([True, False])
        # Enter at frame zero with a pending walk/low-height request. Cancel
        # during settle; request walking again after settle has completed.
        events = []
        if frame in (0, 230):
            events = [{"type": "keyboard", "name": "w", "pressed": True}]
        elif frame == 130:
            events = [{"type": "keyboard", "name": "w", "pressed": False}]
        ctrl = {"KeyboardCtrl": {"keyboard_event": events}}
        _, extras, _, sample_errors = check_observation_and_action(source, student, wrapper, data, ctrl, info)
        errors = np.maximum(errors, sample_errors)
        routes[extras["expert_index"]] += 1
    assert min(routes) > 0

    # Compare nominal physics/actuator/sensor definitions, independently of
    # source reset randomization. This doesn't claim identical random resets.
    source_model = source.get_playback_model()
    target_model = pipeline.env.model
    fields = [
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_pos",
        "body_quat",
        "dof_armature",
        "dof_damping",
        "jnt_range",
        "jnt_axis",
        "jnt_pos",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
        "actuator_ctrllimited",
        "actuator_forcelimited",
        "sensor_type",
        "sensor_objid",
        "sensor_dim",
    ]
    physics = {}
    for field in fields:
        a, b = getattr(source_model, field), getattr(target_model, field)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, err_msg=field)
        physics[field] = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
    collision_fields = [
        "geom_type",
        "geom_bodyid",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_solref",
        "geom_solimp",
        "geom_margin",
        "geom_gap",
    ]

    def collision_ids(model):
        return {model.geom(i).name: i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]}

    source_geoms, target_geoms = collision_ids(source_model), collision_ids(target_model)
    assert source_geoms.keys() == target_geoms.keys()
    for name, left in source_geoms.items():
        right = target_geoms[name]
        for field in collision_fields:
            np.testing.assert_allclose(
                getattr(source_model, field)[left],
                getattr(target_model, field)[right],
                rtol=0,
                atol=1e-12,
                err_msg=f"{name}:{field}",
            )
    physics["named_collision_geoms_equal"] = len(source_geoms)
    return {
        "frames": 340,
        "duplicate_reads": 340,
        "threshold_boundary_cases": boundary_cases,
        "route_counts": routes,
        "max_errors_obs_action_pd_target": errors.tolist(),
        "nominal_model_max_errors": physics,
    }


def verify_physics(source, pipeline):
    """Independent source nominal-model replay of identical PD inputs."""
    model = source.get_playback_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    pipeline.env.reset()
    errors = np.zeros(3)
    for frame in range(120):
        target = np.asarray(pipeline.policy.default_pos) + 0.02 * np.sin(frame * 0.03 + np.arange(29))
        for _ in range(3):
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
        pipeline.env.step(target)
        current = pipeline.env.data
        for index, field in enumerate(("qpos", "qvel", "sensordata")):
            expected, observed = getattr(data, field), getattr(current, field)
            errors[index] = max(errors[index], float(np.max(np.abs(expected - observed))))
            np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-9, err_msg=field)
    return {"control_steps": 120, "physical_substeps": 360, "max_errors_qpos_qvel_sensors": errors.tolist()}


def bounded_episode(source, student, case, seed, output):
    from unilab.envs.locomotion.g1.recovery import recovery_stable_mask

    cfg = g1_unilab_recovery_sim(env=G1UniLabRecoveryEnvCfg(headless=True, reset_supine=case == "supine"), ctrl=[])
    pipeline = UniLabRecoverySimPipeline(cfg)
    env, wrapper = pipeline.env, pipeline.policy
    rng = np.random.default_rng(seed)
    if case == "supine":
        # Source supine reset contract: ±.05 joint noise, clipped initial pose;
        # execution targets themselves remain unmodified.
        env.data.qpos[7:] += rng.uniform(-0.05, 0.05, 29)
        env.data.qpos[7:] = np.clip(env.data.qpos[7:], env.model.jnt_range[1:, 0], env.model.jnt_range[1:, 1])
        mujoco.mj_forward(env.model, env.data)
        env.update()
    wrapper.reset()
    info = new_info(wrapper.gait_phase)
    steps = 150 if case in ("stand", "walk", "low_stand") else (1000 if case == "push" else 750)
    counts = [0, 0, 0]
    errors = np.zeros(3)
    trace = []
    longest = stable_run = 0
    releases = []
    failures = None
    start_x = float(env.data.qpos[0])
    peak_target_excess = 0.0
    force_peak = 0.0
    push_started = None
    try:
        for frame in range(steps):
            # Physical disturbance, no state teleport, reset, or helper force.
            if case == "push":
                body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                env.data.xfrc_applied[body_id, 0] = -600.0 if 100 <= frame < 110 else 0.0
                if frame == 100:
                    push_started = env.recovery_episode_id
                if frame >= 100:
                    assert env.recovery_episode_id == push_started
            events = []
            if frame == 0 and case == "walk":
                events = [{"type": "keyboard", "name": "w", "pressed": True}]
            if case == "low_stand" and frame < 11:
                events = [{"type": "keyboard", "name": "[", "pressed": True}]
            ctrl = {"KeyboardCtrl": {"keyboard_event": events}}
            data = env.get_data()
            was_recovering = wrapper.route.active
            _, extras, pd_target, sample_errors = check_observation_and_action(
                source, student, wrapper, data, ctrl, info
            )
            errors = np.maximum(errors, sample_errors)
            counts[extras["expert_index"]] += 1
            if was_recovering and not wrapper.route.active:
                releases.append(frame)
            limits = env.model.jnt_range[1:]
            peak_target_excess = max(
                peak_target_excess, float(np.max(np.maximum(limits[:, 0] - pd_target, pd_target - limits[:, 1])))
            )
            env.step(pd_target)
            wrapper.post_step_callback([])
            pipeline.safety_check()
            after = env.get_data()
            up = -after.policy_gravity
            height = float(after.recovery_height)
            tilt = float(np.rad2deg(np.arccos(np.clip(up[2], -1, 1))))
            lv, av = float(np.linalg.norm(after.recovery_linvel)), float(np.linalg.norm(after.policy_gyro))
            stable = bool(
                recovery_stable_mask(
                    np.array([after.recovery_height], np.float32),
                    up[None],
                    after.recovery_linvel[None],
                    after.policy_gyro[None],
                    after.recovery_feet_contact[None],
                    source.cfg.recovery,
                )[0]
            )
            stable_run = stable_run + 1 if stable else 0
            longest = max(longest, stable_run)
            force_peak = max(force_peak, float(np.max(np.abs(env.data.actuator_force))))
            trace.append(
                [frame, height, tilt, lv, av, int(stable), extras["expert_index"], extras["recovery_settle_remaining"]]
            )
    except (AssertionError, ValueError, RuntimeError, FloatingPointError) as exc:
        failures = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, AssertionError):
            raise  # never record parity failure as an ordinary policy outcome
    finally:
        env.shutdown()
    np.savez_compressed(output / f"{case}_seed{seed}.npz", trace=np.asarray(trace))
    return {
        "case": case,
        "seed": seed,
        "completed_steps": len(trace),
        "limit_steps": steps,
        "route_counts": counts,
        "recovery_release_frames": releases,
        "ever_stable_1s": longest >= 50,
        "longest_stable_s": longest * 0.02,
        "stable_at_end": stable_run >= 50,
        "final_height_m": float(env.data.qpos[2]),
        "x_displacement_m": float(env.data.qpos[0]) - start_x,
        "max_errors_obs_action_pd_target": errors.tolist(),
        "stop_reason": failures,
        "sampled_50Hz_PD_target_range_excess_rad": peak_target_excess,
        "sampled_50Hz_actuator_force_peak_Nm": force_peak,
        "episode_id": env.recovery_episode_id,
        "physical_push_without_reset": case == "push",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new evidence output directory")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.source_root / "src"))
    from unilab.algos.torch.distill import load_distillation_student_policy
    from unilab.algos.torch.distill.recovery_integration import validate_recovery_student

    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == STUDENT_SHA256
    torch.set_num_threads(1)
    loaded = load_distillation_student_policy(args.checkpoint, device="cpu")
    validate_recovery_student(loaded.policy, loaded.distill_runtime_cfg)
    source = source_environment(args.source_root)
    report = {
        "source_student_sha256": STUDENT_SHA256,
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "protocol": "deterministic I/O then small nominal-physics simulation, no training/hardware",
        "noise_and_domain_randomization": "no observation noise; target nominal physics, no DR; not E132/E136 protocol",
    }
    try:
        probe = UniLabRecoverySimPipeline(g1_unilab_recovery_sim(env=G1UniLabRecoveryEnvCfg(headless=True), ctrl=[]))
        try:
            report["offline_parity"] = verify_offline(source, loaded.policy, probe)
            report["nominal_physics_replay"] = verify_physics(source, probe)
        finally:
            probe.env.shutdown()
        print("OFFLINE_PARITY_PASSED", json.dumps(report["offline_parity"], ensure_ascii=False), flush=True)
        report["episodes"] = []
        for case, seed in [
            ("stand", 0),
            ("low_stand", 0),
            ("walk", 0),
            ("supine", 0),
            ("supine", 1),
            ("supine", 2),
            ("push", 0),
        ]:
            result = bounded_episode(source, loaded.policy, case, seed, args.output)
            report["episodes"].append(result)
            print("EPISODE", json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        source.close()
        (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == STUDENT_SHA256


if __name__ == "__main__":
    main()
