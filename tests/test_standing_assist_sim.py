import unittest
from types import SimpleNamespace

import numpy as np
from box import Box

from robojudo.config.g1.g1_recovery_cfg import g1_unilab_recovery_sim
from robojudo.config.g1.g1_standing_assist_cfg import G1StandingAssistEnvCfg, g1_unilab_standing_assist_sim
from robojudo.pipeline.unilab_standing_assist_sim_pipeline import UniLabStandingAssistSimPipeline


class TestStandingAssistSim(unittest.TestCase):
    def test_off_config_and_mixed_or_real_config_rejected_before_initialization(self):
        self.assertEqual(g1_unilab_recovery_sim().policy.policy_type, "UniLabRecoveryPolicy")
        for env in (SimpleNamespace(is_sim=False), g1_unilab_recovery_sim().env):
            cfg = g1_unilab_standing_assist_sim()
            cfg.env = env
            with self.assertRaises(ValueError):
                UniLabStandingAssistSimPipeline(cfg)

    def test_dryrun_reset_and_cached_reads_preserve_physics_and_controller_time(self):
        pipe = UniLabStandingAssistSimPipeline(
            g1_unilab_standing_assist_sim(env=G1StandingAssistEnvCfg(headless=True), ctrl=[])
        )
        try:
            self.assertIsNone(pipe.policy.core)
            for _ in range(3):
                pipe.step(dry_run=True)
            self.assertIsNone(pipe.policy.core)
            self.assertEqual(pipe.env.recovery_control_frame, 0)
            pipe.step()
            saved = pipe.policy.snapshot_state()
            original = pipe.policy.core.last_frame
            for _ in range(3):
                pipe.step(dry_run=True)
            self.assertEqual(pipe.policy.core.last_frame, original)
            self.assertEqual(pipe.env.recovery_control_frame, 1)
            pipe.policy.reset()
            pipe.policy.restore_state(saved)
            self.assertEqual(pipe.policy.core.last_frame, original)
            data = pipe.env.get_data()
            obs, _ = pipe.policy.get_observation(data, Box(KeyboardCtrl=dict(keyboard_event=[])))
            action = pipe.policy.get_action(obs)
            obs2, _ = pipe.policy.get_observation(data, Box(KeyboardCtrl=dict(keyboard_event=[])))
            np.testing.assert_array_equal(action, pipe.policy.get_action(obs2))
            pipe.env.reborn()
            pipe.policy.get_observation(pipe.env.get_data(), {})
            self.assertEqual(pipe.policy.core.last_frame, 0)
        finally:
            pipe.env.shutdown()


if __name__ == "__main__":
    unittest.main()
