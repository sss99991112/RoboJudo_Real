"""Expose actual MuJoCo sensors with explicit episode/control-frame identity."""

import mujoco
import numpy as np

from robojudo.environment import env_registry
from robojudo.environment.mujoco_env import MujocoEnv


@env_registry.register
class UniLabRecoveryMujocoEnv(MujocoEnv):
    def __init__(self, cfg_env, device="cpu"):
        if not cfg_env.is_sim or cfg_env.actuator_control_mode != "position":
            raise ValueError("recovery environment is simulation-only with source position actuators")
        self.recovery_episode_id = 0
        self.recovery_control_frame = 0
        super().__init__(cfg_env, device)
        self._recovery_velocity = self._resolve_sensor("pelvis_local_linvel")
        self._recovery_feet = [
            [self._resolve_sensor(f"{side}_foot_contact_{i}") for i in range(4)] for side in ("left", "right")
        ]
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id < 0 or self.model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("recovery height currently requires the flat floor scene")
        self._floor_height = float(self.model.geom_pos[floor_id, 2])
        self.reset()

    def _new_episode(self):
        self.recovery_episode_id += 1
        self.recovery_control_frame = 0
        self.data.time = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.update()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        if self.cfg_env.reset_supine:
            self.data.qpos[2] = 0.30
            self.data.qpos[3:7] = [np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0]
        self._new_episode()

    def reborn(self, init_qpos=None):
        if init_qpos is None:
            self.reset()
        else:
            super().reborn(init_qpos)
            self._new_episode()

    def get_data(self):
        data = super().get_data()
        data["recovery_state_source"] = "mujoco_physical_sensors"
        data["recovery_episode_id"] = self.recovery_episode_id
        data["recovery_control_frame"] = self.recovery_control_frame
        data["recovery_height"] = np.float32(self.data.qpos[2] - self._floor_height)
        data["recovery_linvel"] = self._read_sensor(self._recovery_velocity).copy()
        data["recovery_feet_contact"] = np.asarray(
            [any(float(self._read_sensor(sensor)[0]) > 0.5 for sensor in foot) for foot in self._recovery_feet],
            dtype=bool,
        )
        return data

    def step(self, pd_target, hand_pose=None):
        if np.asarray(pd_target).shape != (29,) or not np.isfinite(pd_target).all():
            raise ValueError("recovery PD target must be finite with 29 joints")
        super().step(pd_target, hand_pose)
        self.recovery_control_frame += 1
