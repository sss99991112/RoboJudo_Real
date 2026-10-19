from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from box import Box

from robojudo.config.g1.g1_cfg import g1_unilab_distill
from robojudo.config.g1.g1_recovery_cfg import (
    G1UniLabRecoveryEnvCfg,
    G1UniLabRecoveryPolicyCfg,
    g1_unilab_recovery_sim,
)
from robojudo.environment.unilab_recovery_mujoco_env import UniLabRecoveryMujocoEnv
from robojudo.pipeline.rl_pipeline import PolicyWrapper
from robojudo.pipeline.unilab_recovery_sim_pipeline import UniLabRecoverySimPipeline
from robojudo.policy.recovery_route import RecoveryRoute


def route_update(route, frame, height=0.754, up=(0, 0, 1), velocity=(0, 0, 0), gyro=(0, 0, 0), feet=(1, 1)):
    return route.update(frame=frame, height=height, upvector=up, linvel=velocity, gyro=gyro, feet=feet)


class TestRecoveryRoute(unittest.TestCase):
    def test_inclusive_float32_height_boundary_matches_source(self):
        route = RecoveryRoute(active=True, stable_steps=49, last_frame=0)
        route_update(route, 1, height=float(np.float32(0.65)))
        self.assertFalse(route.active)
        self.assertEqual(route.settle_remaining, 100)
        route = RecoveryRoute(active=True, stable_steps=49, last_frame=0)
        route_update(route, 1, height=np.nextafter(np.float32(0.65), np.float32(0)))
        self.assertTrue(route.active)
        self.assertEqual(route.stable_steps, 0)

    def test_contiguous_window_and_exact_settle_length(self):
        route = RecoveryRoute()
        route_update(route, 0, height=0.30, up=(1, 0, 0))
        for frame in range(1, 50):
            route_update(route, frame)
            route_update(route, frame)  # observing the same frame must not count twice
            self.assertTrue(route.active)
        route_update(route, 50)
        self.assertFalse(route.active)
        self.assertEqual(route.settle_remaining, 100)
        for frame in range(51, 150):
            route_update(route, frame)
            self.assertTrue(route.blocked)
        route_update(route, 150)
        self.assertFalse(route.blocked)

    def test_unstable_frame_restarts_window_and_refall_takes_priority(self):
        route = RecoveryRoute(active=True)
        for frame in range(30):
            route_update(route, frame)
        route_update(route, 30, feet=(1, 0))
        self.assertEqual(route.stable_steps, 0)
        for frame in range(31, 81):
            route_update(route, frame)
        self.assertEqual(route.expert_index([0, 0, 0]), 1)
        route_update(route, 81, up=(0, 0, -1))
        self.assertTrue(route.active)
        self.assertEqual(route.expert_index([0.4, 0, 0]), 2)

    def test_missing_or_nonfinite_sensor_fails_closed(self):
        with self.assertRaises(ValueError):
            route_update(RecoveryRoute(), 0, velocity=(np.nan, 0, 0))
        with self.assertRaises(ValueError):
            route_update(RecoveryRoute(), 0, feet=(1, np.nan))


class TestRecoveryAdapter(unittest.TestCase):
    def setUp(self):
        self.env = UniLabRecoveryMujocoEnv(G1UniLabRecoveryEnvCfg(headless=True))
        self.wrapper = PolicyWrapper(G1UniLabRecoveryPolicyCfg(), self.env.dof_cfg, "cpu")
        self.wrapper.reset()

    def tearDown(self):
        self.env.shutdown()

    def test_recovery_masks_effective_request_but_cancellation_is_preserved(self):
        env = self.env
        env.cfg_env.reset_supine = True
        env.reset()
        press = {"KeyboardCtrl": {"keyboard_event": [{"type": "keyboard", "name": "w", "pressed": True}]}}
        obs, extras = self.wrapper.get_observation(env.get_data(), press)
        self.assertEqual(extras["expert_index"], 2)
        np.testing.assert_array_equal(obs[93:96], [0, 0, 0])
        np.testing.assert_array_equal(obs[97:99], [0, 0])
        self.assertEqual(float(obs[96]), float(np.float32(0.754)))
        self.assertAlmostEqual(float(extras["requested_commands"][0]), 0.4, places=6)
        release = {"KeyboardCtrl": {"keyboard_event": [{"type": "keyboard", "name": "w", "pressed": False}]}}
        self.wrapper.get_observation(env.get_data(), release)
        self.assertFalse(self.wrapper.policy._pressed_motion_keys)
        self.assertEqual(self.wrapper.route.last_frame, 0)

    def test_reset_dryrun_and_reborn_clear_state(self):
        policy = self.wrapper.policy
        obs, _ = self.wrapper.get_observation(self.env.get_data(), {})
        snapshot = policy.snapshot_state()
        self.wrapper.get_pd_target(obs)
        policy.restore_state(snapshot)
        np.testing.assert_array_equal(policy.last_action, np.zeros(29))
        self.assertEqual(policy.route, snapshot["route"])
        self.env.step(self.wrapper.get_pd_target(obs))
        self.wrapper.get_observation(self.env.get_data(), {})
        self.assertEqual(policy.route.last_frame, 1)
        self.env.reborn()
        self.wrapper.get_observation(self.env.get_data(), {})
        self.assertEqual(policy.route.last_frame, 0)
        np.testing.assert_array_equal(policy.last_action, np.zeros(29))

    def test_simulation_producers_and_gait_are_connected(self):
        data = self.env.get_data()
        self.assertEqual(data.recovery_state_source, "mujoco_physical_sensors")
        self.assertEqual(data.recovery_feet_contact.shape, (2,))
        self.assertEqual(data.recovery_linvel.shape, (3,))
        obs, _ = self.wrapper.get_observation(data, {})
        np.testing.assert_allclose(obs[97:99], [0, np.pi])
        target = self.wrapper.get_pd_target(obs)
        np.testing.assert_allclose(target, self.wrapper.last_action + self.wrapper.default_pos, atol=1e-7)
        self.env.step(target)
        obs, _ = self.wrapper.get_observation(self.env.get_data(), {})
        np.testing.assert_allclose(obs[97:99], (np.array([0, np.pi]) + 2 * np.pi * 1.5 * 0.02) % (2 * np.pi), atol=1e-6)
        bad = Box(data)
        del bad["recovery_state_source"]
        with self.assertRaises(ValueError):
            self.wrapper.get_observation(bad, {})

    def test_old_config_remains_off_and_real_env_rejected_before_construction(self):
        old = g1_unilab_distill()
        self.assertEqual(old.pipeline_type, "RlPipeline")
        self.assertEqual(old.policy.policy_type, "UniLabDistillPolicy")
        self.assertEqual(old.env.env_type, "MujocoEnv")
        cfg = g1_unilab_recovery_sim()
        cfg.env = SimpleNamespace(is_sim=False, env_type="UnitreeCppEnv")
        with self.assertRaises(ValueError):
            UniLabRecoverySimPipeline(cfg)

    def test_mixed_imu_or_substeps_rejected_before_environment_construction(self):
        for field, value in [("policy_gyro_sensor", "pelvis_gyro"), ("sim_decimation", 2)]:
            cfg = g1_unilab_recovery_sim()
            setattr(cfg.env, field, value)
            with self.assertRaises(ValueError):
                UniLabRecoverySimPipeline(cfg)

    def test_pipeline_dryrun_does_not_consume_control_time_or_route(self):
        cfg = g1_unilab_recovery_sim(env=G1UniLabRecoveryEnvCfg(headless=True), ctrl=[])
        pipeline = UniLabRecoverySimPipeline(cfg)
        try:
            for _ in range(3):
                pipeline.step(dry_run=True)
            self.assertEqual(pipeline.env.recovery_control_frame, 0)
            self.assertEqual(pipeline.policy.route.last_frame, -1)
            for _ in range(3):
                pipeline.step()
            self.assertEqual(pipeline.env.recovery_control_frame, 3)
        finally:
            pipeline.env.shutdown()


if __name__ == "__main__":
    unittest.main()
