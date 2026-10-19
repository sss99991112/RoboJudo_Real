"""Explicit opt-in CPU simulation adapter for frozen expert assistance."""

import copy
from types import SimpleNamespace

import numpy as np
import torch

from robojudo.policy import policy_registry
from robojudo.policy.standing_assist_compat import CanonicalStandingAssistAdapter
from robojudo.policy.standing_assist_controller import StandingAssistConfig, StandingAssistController
from robojudo.policy.unilab_recovery_policy import UniLabRecoveryPolicy


class ExportHead(torch.nn.Module):
    def __init__(self, model, index):
        super().__init__()
        self.model, self.index = model, index

    def forward(self, obs):
        return self.model(obs, torch.full((len(obs),), self.index, dtype=torch.int64))


@policy_registry.register
class UniLabStandingAssistPolicy(UniLabRecoveryPolicy):
    def __init__(self, cfg_policy, device):
        if str(device) != "cpu":
            raise ValueError("standing-assist experiment is validated only on CPU simulation")
        self.core = None
        self._cached = None
        self._assist_episode = None
        super().__init__(cfg_policy, device)

    def reset(self):
        super().reset()
        self.core = None
        self._cached = None
        self._assist_episode = None

    def snapshot_state(self):
        state = dict(
            base=super().snapshot_state(),
            cached=copy.deepcopy(self._cached),
            episode=self._assist_episode,
            core=None,
            core_ref=self.core,
        )
        if self.core is not None:
            state["core"] = copy.deepcopy(
                {
                    key: getattr(self.core, key)
                    for key in ("info", "assisting", "attempted", "assist_age", "last_frame", "event")
                }
            )
        return state

    def restore_state(self, state):
        super().restore_state(state["base"])
        self._cached, self._assist_episode = copy.deepcopy(state["cached"]), state["episode"]
        if state["core"] is None:
            self.core = None
        else:
            self.core = state["core_ref"]
            for key, value in copy.deepcopy(state["core"]).items():
                setattr(self.core, key, value)

    def get_observation(self, env_data, ctrl_data):
        if (
            env_data.get("recovery_state_source") != "mujoco_physical_sensors"
            or "standing_assist_actuation" not in env_data
        ):
            raise ValueError("standing assist requires its physical simulation environment")
        episode, frame = int(env_data.recovery_episode_id), int(env_data.recovery_control_frame)
        if self._assist_episode != episode:
            self.reset()
            self._assist_episode = episode
        commands = self._get_commands(ctrl_data)
        height = self._get_height_command(ctrl_data)
        if self.core is None:
            contract = env_data.standing_assist_actuation
            names = list(contract["joint_names"])
            order = [names.index(name) for name in self.cfg_action_dof.joint_names]
            actuation = {key: np.asarray(contract[key])[order] for key in ("kp", "kd", "joint_ranges", "force_limits")}
            student = SimpleNamespace(experts=[ExportHead(self._runtime["model"], index) for index in range(3)])
            self.core = StandingAssistController(
                CanonicalStandingAssistAdapter(self.default_pos),
                student,
                StandingAssistConfig(enabled=True),
                actuation=actuation,
            )
        if frame != self.core.last_frame:
            physical = dict(
                height=np.float32(env_data.recovery_height),
                upvector=-np.asarray(env_data.policy_gravity, np.float32),
                gyro=np.asarray(env_data.policy_gyro, np.float32),
                linvel=np.asarray(env_data.recovery_linvel, np.float32),
                feet=np.asarray(env_data.recovery_feet_contact),
                q=np.asarray(env_data.dof_pos, np.float32),
                dq=np.asarray(env_data.dof_vel, np.float32),
            )
            target, extra = self.core.decide(frame, physical, commands=commands, height=height)
            extra["commands"] = self.core.info["recovery_effective_commands"][0].copy()
            extra["target_height"] = self.core.info["recovery_effective_height_commands"][0, 0]
            extra["expert_index"] = extra["expert"]
            extra["recovery_active"] = extra["active"]
            extra["recovery_stable_steps"] = extra["stable_count"]
            extra["recovery_settle_remaining"] = extra["settle"]
            self._cached = (target.copy(), copy.deepcopy(extra))
        extra = copy.deepcopy(self._cached[1])
        extra["requested_commands"], extra["requested_height"] = commands.copy(), height
        self._last_obs = extra["obs"].copy()
        self._selected_expert = extra["expert"]
        return self._last_obs.copy(), extra

    def get_action(self, obs):
        if self._cached is None or not np.array_equal(obs, self._cached[1]["obs"]):
            raise ValueError("current physical observation required before cached action")
        self.last_action = self._cached[1]["action"].copy()
        return self.last_action.copy()
