"""Policy-set actions and action history use each saved policy's coordinates."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim/rl'))
from stage_gates import action_in_env, observation_for_policy, policy_scales


class SwitchScaleTest(unittest.TestCase):
    def test_known_action_preserves_target_angles_and_filter(self):
        action = np.array([.7, -2., .4])
        for policy in (np.array([.3, .3, 1.]), np.array([.5, .5, 1.])):
            common = np.array([.5, .5, 1.])
            converted = np.clip(np.asarray(action_in_env(action, policy, common), float), -1., 1.)
            np.testing.assert_allclose(common*converted, policy*np.clip(action, -1, 1), atol=1e-15)
            previous_angle = np.array([.1, -.1, .2])
            alpha = .53
            np.testing.assert_allclose(common*(previous_angle/common+alpha*(converted-previous_angle/common)),
                                       previous_angle+alpha*(policy*np.clip(action, -1, 1)-previous_angle))

    def test_equal_scales_keep_exact_objects(self):
        scale = np.array([.3, .5])
        action = np.array([2., -.4], dtype=np.float32)
        self.assertIs(action_in_env(action, scale, scale), action)
        raw = np.arange(30, dtype=np.float32)
        env = SimpleNamespace(scale=scale, policy_idx=[0, 1])
        self.assertIs(observation_for_policy(raw, env, scale), raw)

    def test_last_action_is_converted_without_changing_sensors(self):
        env = SimpleNamespace(scale=np.array([.5, .5]), policy_idx=[0, 1], obs_vel=True, active=['a', 'b'])
        raw = np.arange(20, dtype=np.float32)
        raw[16:18] = [.6, -.6]
        result = observation_for_policy(raw, env, np.array([.3, .3]))
        np.testing.assert_allclose(result[16:18], [1., -1.])
        np.testing.assert_array_equal(result[:16], raw[:16])
        np.testing.assert_array_equal(result[18:], raw[18:])
        np.testing.assert_array_equal(raw[16:18], np.array([.6, -.6], dtype=np.float32))

    def test_saved_scale_defaults_and_validation(self):
        self.assertEqual(policy_scales({})['hip_yaw'], .3)
        self.assertEqual(policy_scales({'scale_override': '{"hip_yaw": 0.5}'})['hip_yaw'], .5)
        for v in ('0', '-1', 'NaN'):
            with self.assertRaises(ValueError):
                policy_scales({'scale_override': '{"hip_yaw": '+v+'}'})
