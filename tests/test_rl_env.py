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

    def test_base_velocity_is_the_torso_velocity_in_its_frame(self):
        """Tracking must use the torso's velocity, not the free-joint origin 0.85 m below base_link."""
        from raptor_env import RaptorEnv
        env = RaptorEnv('flat', randomize=False, seed=0)
        env.reset()
        d = env.data
        d.qvel[:] = 0
        d.qvel[0] = .5  # pure forward translation
        mujoco.mj_forward(env.model, d)
        np.testing.assert_allclose(env.body_velocity(env.base)[1], [.5, 0, 0], atol=1e-6)
        d.qvel[:] = 0
        d.qvel[4] = .8  # pitch rate about the root: the torso 0.85 m above moves forward ~0.68 m/s
        mujoco.mj_forward(env.model, d)
        w, v = env.body_velocity(env.base)
        np.testing.assert_allclose(w, [0, .8, 0], atol=1e-6)
        self.assertGreater(v[0], .6)

    def test_metatarsus_counts_as_foot_with_ankle_roll(self):
        """With ankle roll the metatarsus is its own link; touching the ground with it is not a fall."""
        from raptor_env import RaptorEnv
        env = RaptorEnv('flat', dof=12, randomize=False, seed=0)
        env.reset()
        m = env.model
        feet = set(env.foot_geoms['left'])|set(env.foot_geoms['right'])
        meta = {g for g in range(m.ngeom) if 'metatarsus' in m.body(m.geom_bodyid[g]).name}
        self.assertTrue(meta)
        self.assertTrue(meta <= feet)

    def test_ankle_roll_does_not_chatter(self):
        """Light pad on a force-limited velocity servo chattered at +-90 rad/s before the roll armature."""
        from raptor_env import RaptorEnv
        env = RaptorEnv('flat', dof=12, randomize=False, seed=0)
        env.reset()
        roll = [i for i, n in enumerate(env.active) if 'ankle_roll' in n]
        rng, peak = np.random.default_rng(0), 0.
        for _ in range(60):
            action = np.zeros(12)
            action[roll] = rng.normal(0, .11, 2)
            env.step(action)
            peak = max(peak, np.abs(env.data.qvel[env.v_adr][roll]).max())
        self.assertLess(peak, 5.)


if __name__ == '__main__':
    unittest.main()

