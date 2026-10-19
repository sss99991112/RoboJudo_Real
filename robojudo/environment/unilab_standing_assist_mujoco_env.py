"""Physical actuation limits cached once for the explicit simulation variant."""

from robojudo.environment import env_registry
from robojudo.environment.unilab_recovery_mujoco_env import UniLabRecoveryMujocoEnv


@env_registry.register
class UniLabStandingAssistMujocoEnv(UniLabRecoveryMujocoEnv):
    def __init__(self, cfg_env, device="cpu"):
        super().__init__(cfg_env, device)
        model = self.model
        joints = model.actuator_trnid[:, 0]
        self._assist_actuation = dict(
            joint_names=[model.joint(int(j)).name for j in joints],
            kp=model.actuator_gainprm[:, 0].copy(),
            kd=-model.actuator_biasprm[:, 2].copy(),
            joint_ranges=model.jnt_range[joints].copy(),
            force_limits=model.actuator_forcerange.copy(),
        )

    def get_data(self):
        data = super().get_data()
        data["standing_assist_actuation"] = self._assist_actuation
        return data
