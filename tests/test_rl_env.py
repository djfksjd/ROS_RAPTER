"""RaptorEnv contract: 10 position-target actions, proprioceptive obs, exact terrain, fall termination."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sim'/'rl'))
try:
    import gymnasium  # noqa: F401
    import mujoco
    import numpy as np
except ImportError:  # .venv-ai has no simulator stack; use .venv-sim
    mujoco = None


@unittest.skipIf(mujoco is None, 'mujoco/gymnasium not installed')
class RaptorEnvTest(unittest.TestCase):
    def test_spaces_and_actuators(self):
        from raptor_env import RaptorEnv
        env = RaptorEnv('flat', randomize=False, seed=0)
        obs, _ = env.reset()
        self.assertEqual(env.action_space.shape, (10,))
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(env.model.nu, 10)

    def test_targets_stay_inside_joint_ranges(self):
        from raptor_env import RaptorEnv
        env = RaptorEnv('flat', randomize=False, seed=0)
        env.reset()
        for _ in range(5):
            env.step(np.ones(10))
        q = env.data.qpos[env.q_adr]
        self.assertTrue(np.all(q >= env.lo-.05) and np.all(q <= env.hi+.05))

    def test_terrain_surface_matches_height_grid(self):
        import terrain as tr
        from raptor_env import RaptorEnv
        for kind in tr.KINDS:
            env = RaptorEnv(kind, level=1., randomize=False, seed=3)
            env.reset()
            for x in (-1., 0., 1., 2.5):
                geom = np.array([-1], dtype=np.int32)
                dist = mujoco.mj_ray(env.model, env.data, np.array([x, 1., 5.]), np.array([0, 0, -1.]),
                                     None, 1, -1, geom)
                self.assertAlmostEqual(5-dist, tr.height_at(env.heights, x, 1.), delta=.003, msg=f'{kind} x={x}')

    def test_body_contact_ends_episode(self):
        from raptor_env import RaptorEnv
        env = RaptorEnv('flat', randomize=False, seed=0)
        env.reset()
        env.data.qpos[2] -= .5  # sink the body into the ground
        mujoco.mj_forward(env.model, env.data)
        _, reward, terminated, _, _ = env.step(np.zeros(10))
        self.assertTrue(terminated)
        self.assertLess(reward, 0)


if __name__ == '__main__':
    unittest.main()
