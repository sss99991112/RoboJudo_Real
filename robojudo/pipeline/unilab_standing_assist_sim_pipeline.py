from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.unilab_recovery_sim_pipeline import UniLabRecoverySimPipeline


@pipeline_registry.register
class UniLabStandingAssistSimPipeline(UniLabRecoverySimPipeline):
    POLICY_TYPE = "UniLabStandingAssistPolicy"
    ENV_TYPE = "UniLabStandingAssistMujocoEnv"
