"""Simulation recovery contract ported from UniLab's recovery_combined owner.

The network still receives canonical 99-D observations. Physical recovery
state is a separate input to expert selection, never inferred from a command.
"""

from dataclasses import dataclass

import numpy as np

STUDENT_SHA256 = "27d06b8d73734fb2a17c730aa63ff70b2ba4ffbef7d3d71373df43974ab9bf18"
ROLES = {"walk": 0, "stand_height": 1, "recovery": 2}
ROUTING_CONTRACT = {
    "enter_height": 0.40,
    "enter_tilt_deg": 60.0,
    "nominal_settle_steps": 100,
    "standing_height": 0.65,
    "stable_tilt_deg": 10.0,
    "stable_linear_speed": 0.20,
    "stable_angular_speed": 0.30,
    "stable_seconds": 1.0,
    "target_height": 0.754,
    "ctrl_dt": 0.02,
    "state_source": "mujoco_physical_sensors",
    "student_observation_layout": "g1_walk_scaled_99_v1",
    "recovery_teacher_adapter": "g1_recovery_to_walk_99_v1",
}


def require_vector(value, size, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape ({size},)")
    return value


@dataclass
class RecoveryRoute:
    active: bool = False
    stable_steps: int = 0
    settle_remaining: int = 0
    last_frame: int = -1

    def update(self, *, frame, height, upvector, linvel, gyro, feet):
        # Source physics and threshold comparisons are float32. Promoting the
        # measured .65 float32 to a Python float changes the inclusive boundary.
        height = np.asarray(height, dtype=np.float32)
        if height.shape != ():
            raise ValueError("recovery height must be scalar")
        upvector = require_vector(upvector, 3, "upvector")
        linvel = require_vector(linvel, 3, "linvel")
        gyro = require_vector(gyro, 3, "gyro")
        feet = require_vector(feet, 2, "feet")
        if not np.isin(feet, [0, 1]).all() or not np.isfinite(height):
            raise ValueError("recovery requires real boolean contacts and finite height")
        if frame < self.last_frame:
            raise ValueError("recovery frame regressed without an episode reset")
        if frame == self.last_frame:
            return self.active
        c = ROUTING_CONTRACT
        fallen = height < c["enter_height"] or upvector[2] < np.cos(np.deg2rad(c["enter_tilt_deg"]))
        self.active = bool(self.active or fallen)
        stable = (
            height >= c["standing_height"]
            and upvector[2] >= np.cos(np.deg2rad(c["stable_tilt_deg"]))
            and np.linalg.norm(linvel) <= c["stable_linear_speed"]
            and np.linalg.norm(gyro) <= c["stable_angular_speed"]
            and np.all(feet)
        )
        self.stable_steps = self.stable_steps + 1 if self.active and stable else 0
        release = self.active and self.stable_steps >= int(np.ceil(c["stable_seconds"] / c["ctrl_dt"]))
        self.settle_remaining = max(self.settle_remaining - 1, 0)
        if release:
            self.settle_remaining = c["nominal_settle_steps"]
            self.active = False
        self.last_frame = int(frame)
        return self.active

    @property
    def blocked(self):
        return self.active or self.settle_remaining > 0

    def expert_index(self, effective_commands):
        if self.active:
            return ROLES["recovery"]
        commands = require_vector(effective_commands, 3, "effective_commands")
        walking = np.linalg.norm(commands[:2]) > 0.05 or abs(commands[2]) > 0.05
        return ROLES["walk"] if walking else ROLES["stand_height"]
