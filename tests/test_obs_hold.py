"""Stage 1a position-hold observation (RunEnv obs_hold): odometry offset from the zero-command anchor, its mirror."""
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

KW = dict(arch8=True, real_mass=True, randomize=False, seed=1, model_rev=3, cmd_max=[0., 0., 0.])


@unittest.skipIf(mujoco is None, 'mujoco/gymnasium not installed')
class ObsHoldTest(unittest.TestCase):
    def test_offset_in_heading_frame(self):
        from run_env import RunEnv
        base = RunEnv(**KW)
        e = RunEnv(**KW, obs_hold={'noise': 0., 'walk': 0.})
        self.assertEqual(e.observation_space.shape[0], base.observation_space.shape[0]+3)
        o, _ = e.reset(seed=1)
        np.testing.assert_allclose(o[-3:], 0., atol=1e-6)
        e.data.qpos[0] += .2  # move the base 0.2 m along world x; heading ~0 at reset
        mujoco.mj_forward(e.model, e.data)
        e.steps += 1
        v = e._obs()[-3:]
        self.assertAlmostEqual(float(v[0]), .2, delta=.02)
        self.assertAlmostEqual(float(v[1]), 0., delta=.02)

    def test_zero_while_commanded(self):
        from run_env import RunEnv
        e = RunEnv(**{**KW, 'cmd_max': [1., 0., 0.]}, obs_hold={'noise': .02, 'walk': .005})
        e.reset(seed=1)
        e.command = np.array([.5, 0., 0.])
        e.steps += 1
        np.testing.assert_array_equal(e._obs()[-3:], 0.)
        self.assertIsNone(e.hold_anchor)

    def test_mirror(self):
        from mirror_probe import ObservationReflection
        from run_env import RunEnv
        e = RunEnv(**KW, obs_hold={'noise': 0., 'walk': 0.})
        o, _ = e.reset(seed=1)
        r = ObservationReflection(e.active, e.policy_idx, e.q0, e.obs_vel, e.obs_contact, 0, True)
        x = o + np.random.default_rng(0).normal(0, .1, o.shape)
        y = r.observation(x)
        np.testing.assert_allclose(y[-3:], x[-3:]*[1, -1, -1])
        np.testing.assert_allclose(r.observation(y), x, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
