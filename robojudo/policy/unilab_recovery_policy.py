"""Frozen UniLab three-expert student; simulation-only physical routing."""

from __future__ import annotations

import copy
import json

import numpy as np
import torch

from robojudo.policy import policy_registry
from robojudo.policy.recovery_route import (
    ROLES,
    ROUTING_CONTRACT,
    STUDENT_SHA256,
    RecoveryRoute,
    require_vector,
)
from robojudo.policy.unilab_distill_policy import UniLabDistillPolicy


@policy_registry.register
class UniLabRecoveryPolicy(UniLabDistillPolicy):
    def __init__(self, cfg_policy, device):
        if (
            cfg_policy.freq != 50
            or cfg_policy.gait_frequency != 1.5
            or cfg_policy.expected_obs_dim != 99
            or cfg_policy.expected_action_dim != 29
            or cfg_policy.action_scale != 1.0
            or cfg_policy.action_beta != 1.0
            or cfg_policy.action_clip is not None
            or cfg_policy.command_xy_threshold != 0.05
            or cfg_policy.command_yaw_threshold != 0.05
            or cfg_policy.default_target_height != 0.754
        ):
            raise ValueError("three-expert migration requires the frozen source control contract")
        self.route = RecoveryRoute()
        self._episode_id = None
        self._phase_frame = -1
        self._selected_expert = None
        super().__init__(cfg_policy, device)
        self.default_dof_pos = np.asarray(self.default_dof_pos, dtype=np.float32)
        self.default_pos = np.asarray(self.default_pos, dtype=np.float32)

    def _build_runtime(self, policy_file):
        extra = {"contract.json": ""}
        model = torch.jit.load(policy_file, map_location=self.device, _extra_files=extra).eval()
        metadata = json.loads(extra["contract.json"])
        expected = {
            "obs_dim": 99,
            "action_dim": 29,
            "roles": ROLES,
            "routing": ROUTING_CONTRACT,
            "source_sha256": STUDENT_SHA256,
            "inputs": ["obs", "expert_index"],
        }
        if metadata != expected:
            raise ValueError("three-expert TorchScript contract metadata mismatch")
        return {"kind": "torchscript", "model": model}

    def _check_runtime_contract(self):
        probe = torch.zeros((1, 99), device=self.device)
        for index in range(3):
            with torch.no_grad():
                result = self._runtime["model"](probe, torch.tensor([index], device=self.device))
            if tuple(result.shape) != (1, 29) or not torch.isfinite(result).all():
                raise ValueError("invalid three-expert TorchScript output")

    def reset(self):
        super().reset()
        self.route = RecoveryRoute()
        self._episode_id = None
        self._phase_frame = -1
        self._selected_expert = None

    def snapshot_state(self):
        return copy.deepcopy(
            {
                "base": super().snapshot_state(),
                "route": self.route,
                "episode": self._episode_id,
                "phase_frame": self._phase_frame,
                "expert": self._selected_expert,
                "pressed": self._pressed_motion_keys,
                "height": self.target_height,
            }
        )

    def restore_state(self, state):
        state = copy.deepcopy(state)
        super().restore_state(state["base"])
        self.route = state["route"]
        self._episode_id = state["episode"]
        self._phase_frame = state["phase_frame"]
        self._selected_expert = state["expert"]
        self._pressed_motion_keys = state["pressed"]
        self.target_height = state["height"]

    def get_observation(self, env_data, ctrl_data):
        if env_data.get("recovery_state_source") != "mujoco_physical_sensors":
            raise ValueError("three-expert policy requires the dedicated MuJoCo recovery environment")
        episode = int(env_data.recovery_episode_id)
        frame = int(env_data.recovery_control_frame)
        if self._episode_id != episode:
            self.reset()
            self._episode_id = episode
        gyro = require_vector(env_data.policy_gyro, 3, "torso gyro")
        gravity = require_vector(env_data.policy_gravity, 3, "negative torso upvector")
        commands = self._get_commands(ctrl_data)
        height_request = self._get_height_command(ctrl_data)

        # Source apply_action advances raw phase before the next observation.
        # With recovery_combined's default gait_constraint disabled, standing
        # advances too. Recovery _update_commands resets raw phase to zero.
        if frame != self._phase_frame:
            if self._phase_frame >= 0:
                delta = frame - self._phase_frame
                self.gait_phase = (self.gait_phase + delta * 2.0 * np.pi * self.gait_frequency * self.dt) % (
                    2.0 * np.pi
                )
            self.route.update(
                frame=frame,
                height=float(env_data.recovery_height),
                upvector=-gravity,
                linvel=env_data.recovery_linvel,
                gyro=gyro,
                feet=env_data.recovery_feet_contact,
            )
            if self.route.active:
                self.gait_phase[:] = 0.0
            self._phase_frame = frame

        effective_commands = np.zeros(3, np.float32) if self.route.blocked else commands.copy()
        effective_height = np.float32(0.754) if self.route.blocked else height_request
        dof_pos = require_vector(env_data.dof_pos, 29, "dof_pos")
        dof_vel = require_vector(env_data.dof_vel, 29, "dof_vel")
        obs = np.concatenate(
            [
                gyro * 0.25,
                gravity,
                (dof_pos - np.asarray(self.default_dof_pos, np.float32)).astype(np.float32),
                dof_vel * 0.05,
                np.asarray(self.last_action, np.float32),
                effective_commands,
                np.asarray([effective_height], np.float32),
                self.gait_phase,
            ]
        ).astype(np.float32)
        if obs.shape != (99,) or not np.isfinite(obs).all():
            raise ValueError("invalid canonical three-expert observation")
        self._selected_expert = self.route.expert_index(effective_commands)
        self._last_obs = obs.copy()
        return obs, {
            "commands": effective_commands,
            "requested_commands": commands.copy(),
            "target_height": effective_height,
            "requested_height": height_request,
            "gait_phase": self.gait_phase.copy(),
            "expert_index": self._selected_expert,
            "recovery_active": self.route.active,
            "recovery_stable_steps": self.route.stable_steps,
            "recovery_settle_remaining": self.route.settle_remaining,
        }

    def get_action(self, obs):
        if self._selected_expert is None:
            raise RuntimeError("observation and physical route required before inference")
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        index = torch.tensor([self._selected_expert], dtype=torch.int64, device=self.device)
        with torch.inference_mode():
            action = self._runtime["model"](tensor, index).cpu().numpy()[0].copy()
        action = require_vector(action, 29, "action")
        self.last_action = action.copy()
        return action

    def post_step_callback(self, commands=None):
        # Frame ownership is in get_observation; dry runs and repeated reads
        # cannot advance route windows or gait.
        pass
