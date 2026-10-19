"""One opt-in simulation owner; no hardware variant is registered."""

from typing import Literal

from robojudo.config import cfg_registry
from robojudo.config.g1.env.g1_mujuco_env_cfg import G1UniLabMujocoEnvCfg
from robojudo.config.g1.policy.g1_unilab_policy_cfg import (
    G1UniLabDistillDoF,
    G1UniLabDistillPolicyCfg,
)
from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg


class G1UniLabRecoveryPolicyCfg(G1UniLabDistillPolicyCfg):
    policy_type: Literal["UniLabRecoveryPolicy"] = "UniLabRecoveryPolicy"
    policy_name: str = "g1_three_expert_dagger_iteration_2"
    exported_policy_file: str | None = None
    height_up_key: str = "]"
    height_down_key: str = "["

    @property
    def policy_file(self):
        return self.exported_policy_file or super().policy_file


class G1UniLabRecoveryEnvCfg(G1UniLabMujocoEnvCfg):
    env_type: Literal["UniLabRecoveryMujocoEnv"] = "UniLabRecoveryMujocoEnv"
    is_sim: Literal[True] = True
    dof: G1UniLabDistillDoF = G1UniLabDistillDoF()
    born_place_align: bool = False
    visualize_extras: bool = False
    headless: bool = False
    reset_supine: bool = False


@cfg_registry.register
class g1_unilab_recovery_sim(RlPipelineCfg):
    robot: str = "g1"
    pipeline_type: Literal["UniLabRecoverySimPipeline"] = "UniLabRecoverySimPipeline"
    env: G1UniLabRecoveryEnvCfg = G1UniLabRecoveryEnvCfg()
    policy: G1UniLabRecoveryPolicyCfg = G1UniLabRecoveryPolicyCfg()
    ctrl: list[KeyboardCtrlCfg] = [KeyboardCtrlCfg()]
    do_safety_check: Literal[True] = True
