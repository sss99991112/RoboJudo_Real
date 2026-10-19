"""Explicit experimental simulation owner; old configurations remain unchanged."""

from typing import Literal

from robojudo.config import cfg_registry
from robojudo.config.g1.g1_recovery_cfg import G1UniLabRecoveryEnvCfg, G1UniLabRecoveryPolicyCfg, g1_unilab_recovery_sim


class G1StandingAssistPolicyCfg(G1UniLabRecoveryPolicyCfg):
    policy_type: Literal["UniLabStandingAssistPolicy"] = "UniLabStandingAssistPolicy"


class G1StandingAssistEnvCfg(G1UniLabRecoveryEnvCfg):
    env_type: Literal["UniLabStandingAssistMujocoEnv"] = "UniLabStandingAssistMujocoEnv"


@cfg_registry.register
class g1_unilab_standing_assist_sim(g1_unilab_recovery_sim):
    pipeline_type: Literal["UniLabStandingAssistSimPipeline"] = "UniLabStandingAssistSimPipeline"
    env: G1StandingAssistEnvCfg = G1StandingAssistEnvCfg()
    policy: G1StandingAssistPolicyCfg = G1StandingAssistPolicyCfg()
