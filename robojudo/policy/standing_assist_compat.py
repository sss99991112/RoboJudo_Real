"""Local numerical boundaries for the frozen UniLab standing-assist controller."""

from types import SimpleNamespace

import numpy as np

from robojudo.policy.recovery_route import ROUTING_CONTRACT, RecoveryRoute


def validate_recovery_env(env):
    for key in ("enter_height", "enter_tilt_deg", "nominal_settle_steps"):
        if getattr(env.cfg.recovery_routing, key) != ROUTING_CONTRACT[key]:
            raise ValueError(f"routing contract mismatch: {key}")
    for key in ("standing_height", "stable_tilt_deg", "stable_linear_speed", "stable_angular_speed", "stable_seconds"):
        if getattr(env.cfg.recovery, key) != ROUTING_CONTRACT[key]:
            raise ValueError(f"stability contract mismatch: {key}")
    if env.cfg.ctrl_dt != 0.02 or env.cfg.commands.default_height != 0.754:
        raise ValueError("control period/target height mismatch")


def update_recovery_route(info, *, height, gravity, linvel, gyro, feet, steps, **_):
    if "recovery_active" not in info:
        info.update(
            recovery_active=np.zeros(1, bool),
            recovery_route_stable_steps=np.zeros(1, np.int32),
            recovery_route_last_step=np.full(1, -1, np.int64),
            recovery_settle_remaining=np.zeros(1, np.int32),
        )
    route = RecoveryRoute(
        active=bool(info["recovery_active"][0]),
        stable_steps=int(info["recovery_route_stable_steps"][0]),
        settle_remaining=int(info["recovery_settle_remaining"][0]),
        last_frame=int(info["recovery_route_last_step"][0]),
    )
    route.update(
        frame=int(steps[0]), height=height[0], upvector=gravity[0], linvel=linvel[0], gyro=gyro[0], feet=feet[0]
    )
    for key, value in (
        ("recovery_active", route.active),
        ("recovery_route_stable_steps", route.stable_steps),
        ("recovery_route_last_step", route.last_frame),
        ("recovery_settle_remaining", route.settle_remaining),
    ):
        info[key][:] = value
    return info["recovery_active"]


def recovery_expert_indices(info):
    commands = info["recovery_effective_commands"][0]
    return np.asarray(
        [
            2
            if info["recovery_active"][0]
            else 0
            if np.linalg.norm(commands[:2]) > 0.05 or abs(commands[2]) > 0.05
            else 1
        ],
        np.int64,
    )


class CanonicalStandingAssistAdapter:
    """99-D actor and float32 target arithmetic, without a UniLab dependency."""

    num_envs = 1

    def __init__(self, default):
        self.default_angles = np.asarray(default, np.float32).copy()
        self.cfg = SimpleNamespace(
            recovery=SimpleNamespace(
                **{
                    key: ROUTING_CONTRACT[key]
                    for key in (
                        "standing_height",
                        "stable_tilt_deg",
                        "stable_linear_speed",
                        "stable_angular_speed",
                        "stable_seconds",
                    )
                }
            ),
            recovery_routing=SimpleNamespace(
                **{key: ROUTING_CONTRACT[key] for key in ("enter_height", "enter_tilt_deg", "nominal_settle_steps")}
            ),
            commands=SimpleNamespace(default_height=0.754),
            ctrl_dt=0.02,
        )

    def _compute_obs(self, info, linvel, gyro, upvector, q, dq):
        blocked = info["recovery_active"] | (info["recovery_settle_remaining"] > 0)
        commands, heights = info["commands"].copy(), info["height_commands"].copy()
        commands[blocked] = 0
        heights[blocked] = np.float32(0.754)
        info["recovery_effective_commands"] = commands
        info["recovery_effective_height_commands"] = heights
        return {
            "obs": np.concatenate(
                [
                    gyro * 0.25,
                    -upvector,
                    q - self.default_angles,
                    dq * 0.05,
                    info["current_actions"],
                    commands,
                    heights,
                    info["gait_phase"],
                ],
                axis=1,
            ).astype(np.float32)
        }

    def apply_action(self, action, state):
        info = state.info
        info["last_actions"] = info.get("current_actions", np.zeros_like(action))
        info["current_actions"] = action
        phase = info["gait_phase"]
        delta = float(2 * np.pi * 1.5 * 0.02)
        phase[:, 0] = (phase[:, 0] + delta) % (2 * np.pi)
        phase[:, 1] = (phase[:, 1] + delta) % (2 * np.pi)
        info["gait_phase"] = phase
        info["executed_actions"] = action
        return action + self.default_angles
