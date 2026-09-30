"""RecoverEnv (T1 get-up) contract: fallen start, no body-contact termination, success when standing."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sim'/'rl'))
try:
    import mujoco
    import numpy as np
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, 'mujoco not installed')
class RecoverEnvTest(unittest.TestCase):
    def test_starts_fallen_and_survives_body_contact(self):
        from recover_env import RecoverEnv
        env = RecoverEnv(randomize=False, seed=0)
        for mode in ('side', 'back', 'front', 'random'):
            obs, info = env.reset(options={'mode': mode})
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertGreater(env.data.ncon, 0)
            _, _, term, _, info = env.step(np.zeros(12))
            self.assertGreater(info['tilt'], 1.)
            self.assertFalse(term)

    def test_success_when_held_standing(self):
        from recover_env import RecoverEnv
        env = RecoverEnv(randomize=False, seed=0, stand_hold_s=.1)
        env.reset(options={'mode': 'side'})
        m, d = env.model, env.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.q_adr(env)] = env.q0
        mujoco.mj_forward(m, d)
        lowest = min(d.geom_xpos[g][2]-.02 for s in env.foot_geoms for g in env.foot_geoms[s])
        d.qpos[2] += -lowest+.005
        d.ctrl[env.act] = env.q0
        mujoco.mj_forward(m, d)
        done = False
        for _ in range(10):
            _, r, done, _, info = env.step(np.zeros(12))
            if done:
                break
        self.assertTrue(info['success'])

    @staticmethod
    def q_adr(env):
        return env.q_adr


if __name__ == '__main__':
    unittest.main()
