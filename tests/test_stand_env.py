"""Standing reward priorities and preservation of the source dynamics."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim/rl'))
try:
    import mujoco
    import numpy as np
    from run_env import RunEnv
    from stand_env import StandEnv, stand_reward_terms
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, 'simulator stack unavailable')
class StandEnvTest(unittest.TestCase):
    def test_terminal_penalty_changes_only_terminal_reward(self):
        kwargs=dict(arch8=True,real_mass=True,hip_back=.2,randomize=False,seed=7,
                    springs_override={'knee_pitch':[170,.956,'bi','latch']})
        a,b=StandEnv(**kwargs),StandEnv(**kwargs,fall_penalty=100.)
        try:
            for e in [a,b]:
                e.init_seed=22001;e.reset()
            terminated=False
            for _ in range(100):
                x,y=a.step(np.zeros(8)),b.step(np.zeros(8))
                np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
                np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
                np.testing.assert_array_equal(x[0],y[0])
                self.assertEqual(x[2:4],y[2:4])
                self.assertAlmostEqual(y[1]-x[1],-90. if x[2] else 0.,places=12)
                if x[2]:
                    terminated=True;break
            self.assertTrue(terminated)
        finally:
            a.close();b.close()
        for penalty in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                StandEnv(fall_penalty=penalty)

    def test_position_and_flight_are_worse_than_stationary_support(self):
        def reward(**changes):
            values = dict(displacement=np.zeros(2), velocity=np.zeros(3), tilt=0.,
                          angular_velocity=np.zeros(3), height_error=0., flight=False,
                          unloaded_fraction=0., heat=np.zeros(8), action_change=np.zeros(8))
            values.update(changes)
            return sum(stand_reward_terms(**values).values())
        self.assertGreater(reward(), reward(displacement=np.array([.2,0.])))
        self.assertGreater(reward(), reward(flight=True, unloaded_fraction=1.))
        self.assertGreater(reward(), reward(velocity=np.array([1.,0.,0.])))
        self.assertGreater(reward(), reward(heat=np.full(8,2.)))

    def test_reward_replacement_preserves_original_physical_trajectory(self):
        kwargs = dict(arch8=True, real_mass=True, hip_back=.2,
                      springs_override={'knee_pitch':[170,.956,'bi','latch']},
                      randomize=False, seed=7)
        base, stand = RunEnv(**kwargs), StandEnv(**kwargs)
        try:
            for env in [base, stand]:
                env.init_seed=22001
                env.reset()
                env.command=np.zeros(3)
                env.resample_steps=0
            np.testing.assert_array_equal(base.data.qpos, stand.data.qpos)
            np.testing.assert_array_equal(base.data.qvel, stand.data.qvel)
            self.assertEqual(stand.action_space.shape, (8,))
            self.assertEqual(base.observation_space.shape, stand.observation_space.shape)
            for i in range(12):
                action=np.linspace(-.03,.03,8)*(-1)**i
                a,b=base.step(action),stand.step(action)
                np.testing.assert_array_equal(base.data.qpos,stand.data.qpos)
                np.testing.assert_array_equal(base.data.qvel,stand.data.qvel)
                np.testing.assert_array_equal(a[0],b[0])
                self.assertEqual(a[2:4],b[2:4])
                self.assertTrue(np.isfinite(b[1]))
                self.assertNotIn('gait',b[4]['stand_terms'])
            stand.command[0]=4.
            with self.assertRaises(ValueError):
                stand.step(np.zeros(8))
        finally:
            base.close()
            stand.close()
