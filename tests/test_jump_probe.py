"""Evidence 199: contact timing, contiguous landing hold, and passive instrumentation."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT/'docs/evidence/199-l80-jump'
sys.path.insert(0, str(DIR))
from jump_probe import (Recorder, measurements, make, encode, GUESS, keyframes, action_at, reset_stationary)
from terrain_probe import add_edge
import terrain as tr
from unittest.mock import patch


class JumpProbeTest(unittest.TestCase):
    def setUp(self):
        self.terrain_size = tr.SIZE

    def tearDown(self):
        tr.SIZE = self.terrain_size

    def fixture(self):
        n = 2000
        a = dict(t=np.arange(1, n+1)*.001, xyz=np.tile([0., 0., .75], (n, 1)),
                 feet_force_n=np.full((n, 2), 50.), feet_n=np.ones((n, 2), int),
                 velocity=np.zeros((n, 3)), tilt=np.zeros(n), body_hit=np.zeros(n, bool))
        a['feet_n'][300:400] = 0; a['feet_force_n'][300:400] = 0
        a['velocity'][300, 2] = .2; a['xyz'][350, 2] = .8
        a['feet_force_n'][400] = [220., 180.]
        return SimpleNamespace(weight=100., h_nom=.75), a

    def test_one_ms_flight_and_impact(self):
        env, a = self.fixture(); r = measurements(env, a, None)
        self.assertAlmostEqual(r['event']['flight_s'], .1)
        self.assertAlmostEqual(r['event']['apex_rise_m'], .05)
        self.assertEqual(r['landing_force_peak_n'], 400.)
        self.assertAlmostEqual(r['stand_after_landing_s'], 1.6)

    def test_contacts_with_zero_force_are_not_flight(self):
        env, a = self.fixture(); a['feet_n'][:] = 1
        self.assertIsNone(measurements(env, a, None)['event'])

    def test_initial_air_and_one_foot_air_are_not_jump(self):
        env, a = self.fixture(); a['feet_force_n'][:300] = 0
        self.assertIsNone(measurements(env, a, None)['event'])
        env, a = self.fixture(); a['feet_n'][300:400, 1] = 1
        self.assertIsNone(measurements(env, a, None)['event'])

    def test_standing_must_be_contiguous_and_fall_is_not_survival(self):
        env, a = self.fixture(); a['velocity'][1000:1100, 0] = 1.
        r = measurements(env, a, 2.)
        self.assertAlmostEqual(r['stand_after_landing_s'], .9)
        self.assertEqual(r['held_after_landing_without_fall_s'], 0.)

    def test_recorder_does_not_change_step_or_rng(self):
        e1, e2 = make(), make()
        try:
            e1.reset(seed=0); e2.reset(seed=0)
            e1.resample_steps = e2.resample_steps = 0
            times, values = keyframes(e1, encode(GUESS))
            original_after, original_clamp = e2.rz.after_substep, e2._clamp
            with Recorder(e2) as rec:
                for k in range(25):
                    action = action_at(k*.02, times, values)
                    r1, r2 = e1.step(action), e2.step(action)
                    np.testing.assert_array_equal(r1[0], r2[0])
                    self.assertEqual(r1[1:4], r2[1:4])
                    np.testing.assert_array_equal(e1.data.qpos, e2.data.qpos)
                    np.testing.assert_array_equal(e1.data.qvel, e2.data.qvel)
                self.assertEqual(len(rec.rows['t']), 500)
            self.assertEqual(e2.rz.after_substep, original_after)
            self.assertEqual(e2._clamp, original_clamp)
            self.assertEqual(e1.rng.bit_generator.state, e2.rng.bit_generator.state)
        finally:
            e1.close(); e2.close()

    def test_pose_jitter_has_no_initial_root_velocity(self):
        env = make()
        try:
            reset_stationary(env, 9100)
            np.testing.assert_array_equal(env.data.qvel[:6], np.zeros(6))
            self.assertGreater(np.linalg.norm(env.data.qpos[env.q_adr]-env.q0), 0.)
        finally: env.close()

    def test_recorder_restores_hooks_after_error(self):
        env = make()
        try:
            env.reset(); original = env._clamp
            with self.assertRaises(RuntimeError):
                with Recorder(env): raise RuntimeError('probe failure')
            self.assertEqual(env._clamp, original)
        finally: env.close()

    def test_edge_height_and_ground_query_match_frozen_model(self):
        for height in (.1, -.1):
            def builder(spec, kind, level, rng):
                return add_edge(spec, kind, level, rng, height=height)
            with patch.object(tr, 'add_terrain', builder):
                env = make(9200, terrain={'kinds': {'boxes': 1}, 'max_level': 0., 'size': 40.})
                try:
                    env.reset(seed=9200)
                    self.assertEqual(env.kind, 'boxes')
                    self.assertEqual(tr.height_at(env.heights, 0., 0.), 0.)
                    self.assertEqual(tr.height_at(env.heights, 3., 0.), height)
                    self.assertEqual(env.ground_z(), 0.)
                finally: env.close()

if __name__ == '__main__': unittest.main()
