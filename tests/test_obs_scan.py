"""Ground-height scan observation (RunEnv obs_scan), boxes terrain, and continuing a policy with the scan appended."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sim'/'rl'))
try:
    import gymnasium  # noqa: F401
    import mujoco  # noqa: F401
    import numpy as np
except ImportError:  # .venv-ai has no simulator stack; use .venv-sim
    mujoco = None

DX = [-.3, 0., .3, .6, .9, 1.2]


@unittest.skipIf(mujoco is None, 'mujoco/gymnasium not installed')
class ObsScanTest(unittest.TestCase):
    def test_boxes_heights(self):
        import terrain as tr
        h = tr.heights('boxes', 1., np.random.default_rng(0))
        n = h.shape[0]
        x = np.linspace(-tr.SIZE/2, tr.SIZE/2, n)
        self.assertTrue(np.all(h[:, x < 1.5] == 0.))  # flat spawn pad
        steps = np.diff(h[0])[np.diff(h[0]) != 0]
        self.assertTrue(np.all(steps <= tr.BOX_UP+1e-9) and np.all(steps >= -tr.BOX_DOWN-1e-9))
        self.assertTrue(np.all(h == h[0]))  # constant across y

    def test_scan_appended_and_flat_zero(self):
        from run_env import RunEnv
        kw = dict(arch8=True, real_mass=True, model_rev=2, randomize=False, seed=1)
        a, _ = RunEnv(**kw).reset(seed=1)
        env = RunEnv(**kw, obs_scan={'dx': DX})
        b, _ = env.reset(seed=1)
        self.assertEqual(env.observation_space.shape[0], len(a)+len(DX))
        np.testing.assert_allclose(b[:len(a)], a)
        np.testing.assert_allclose(b[len(a):], 0., atol=1e-6)  # flat ground at the spawn height

    def test_scan_sees_ledge_and_delay(self):
        from run_env import RunEnv
        env = RunEnv(arch8=True, real_mass=True, model_rev=2, randomize=False, seed=1, obs_scan={'dx': DX, 'delay': 2})
        env.reset(seed=1)
        env.heights = np.where(np.linspace(-4, 4, env.heights.shape[0])[None, :] >= .5, .2, 0.)*np.ones_like(env.heights)
        first = env._scan().copy()  # same control step as the reset reading: cached
        env.steps += 1; second = env._scan().copy()
        env.steps += 1; held = env._scan().copy()
        env.steps += 1; third = env._scan().copy()  # the reading of step 1, two steps back
        np.testing.assert_allclose(first, 0., atol=1e-6)
        np.testing.assert_allclose(second, first)
        np.testing.assert_allclose(held, first)  # step 2 returns the reading of step 0
        np.testing.assert_allclose(third[np.array(DX) >= .6], .2, atol=.02)
        np.testing.assert_allclose(third[np.array(DX) <= 0.], 0., atol=.02)

    def test_mirror_keeps_scan(self):
        from mirror_probe import ObservationReflection
        from run_env import RunEnv
        env = RunEnv(arch8=True, real_mass=True, model_rev=2, randomize=False, seed=1, obs_scan={'dx': DX})
        o, _ = env.reset(seed=1)
        r = ObservationReflection(env.active, env.policy_idx, env.q0, env.obs_vel, env.obs_contact, len(DX))
        self.assertEqual(r.size, len(o))
        o = o+np.r_[np.zeros(len(o)-len(DX)), np.arange(len(DX))]
        np.testing.assert_allclose(r.observation(o)[-len(DX):], o[-len(DX):])


if __name__ == '__main__':
    unittest.main()
