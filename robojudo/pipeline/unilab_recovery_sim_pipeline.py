"""Recovery-aware protection scoped to the dedicated simulation entrypoint."""

import numpy as np

from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.rl_pipeline import RlPipeline


@pipeline_registry.register
class UniLabRecoverySimPipeline(RlPipeline):
    POLICY_TYPE = "UniLabRecoveryPolicy"
    ENV_TYPE = "UniLabRecoveryMujocoEnv"

    def __init__(self, cfg):
        # Validate before constructing any environment or controller.
        if (
            not cfg.env.is_sim
            or cfg.env.env_type != self.ENV_TYPE
            or cfg.policy.policy_type != self.POLICY_TYPE
            or cfg.env.policy_gyro_sensor != "torso_gyro"
            or cfg.env.policy_upvector_sensor != "torso_upvector"
            or cfg.env.actuator_control_mode != "position"
            or cfg.env.sim_decimation != 3
            or not np.isclose(cfg.env.sim_dt, 0.02 / 3, atol=1e-12, rtol=0)
            or not cfg.do_safety_check
            or not np.isclose(cfg.env.sim_dt * cfg.env.sim_decimation, 0.02, atol=1e-12, rtol=0)
        ):
            raise ValueError("three-expert pipeline requires its complete simulation-only configuration")
        super().__init__(cfg)

    def safety_check(self):
        data = self.env.get_data()
        values = [data.dof_pos, data.dof_vel, data.policy_gyro, data.policy_gravity, data.recovery_linvel]
        if not all(np.isfinite(value).all() for value in values):
            raise FloatingPointError("non-finite recovery simulation state")
        # Source G1RecoveryCombined termination limits. Tilt is routed into
        # recovery instead of triggering the ordinary pipeline's reborn.
        if np.linalg.norm(data.recovery_linvel) > 20.0 or np.max(np.abs(data.dof_vel)) > 40.0:
            raise RuntimeError("source recovery simulation velocity termination boundary reached")

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        super().post_step_callback(env_data, ctrl_data, extras, pd_target)
        if "[SHUTDOWN]" in ctrl_data.get("COMMANDS", []):
            raise SystemExit("recovery simulation stopped")
