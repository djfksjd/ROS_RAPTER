"""RunEnv (T1) contract: mass scaling, dead-band springs, actuator torque/speed/power clamps, tail lock, momentum log."""
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
class RunEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from run_env import RunEnv
        cls.env = RunEnv(randomize=False, seed=0)
        cls.env.reset()

    def test_spaces_mass_and_springs(self):
        env, m = self.env, self.env.model
        self.assertEqual(env.action_space.shape, (12,))
        self.assertEqual(env.reset()[0].shape, env.observation_space.shape)
        self.assertAlmostEqual(m.body_subtreemass[env.root], 5., delta=.01)
        self.assertEqual(m.ntendon, 6)  # 3 springs x 2 legs, case (c)

    def test_unilateral_springs_have_a_dead_band(self):
        env, m, d = self.env, self.env.model, self.env.data
        env.reset()
        hip = m.joint('left_hip_pitch_joint'); ankle = m.joint('left_ankle_pitch_joint')
        for joint, q, expect in ((hip, -.5, 0.), (hip, .2, -130*(.2-.041)), (ankle, -1., 0.), (ankle, -1.5, 57.2*(1.5-1.215))):
            d.qpos[m.jnt_qposadr[joint.id]] = q
            mujoco.mj_forward(m, d)
            self.assertAlmostEqual(d.qfrc_passive[m.jnt_dofadr[joint.id]], expect, delta=.3, msg=joint.name)

    def test_force_range_follows_speed_and_power(self):
        env, d = self.env, self.env.data
        env.reset()
        i = env.active.index('left_hip_pitch_joint')  # 35 N·m, 27 rad/s, 350 W
        d.qvel[env.v_adr[i]] = 20.
        lo, hi = env._clamp()
        self.assertAlmostEqual(hi[i], 350/20); self.assertAlmostEqual(lo[i], -350/20)
        d.qvel[env.v_adr[i]] = 30.  # beyond the speed limit: no driving torque
        lo, hi = env._clamp()
        self.assertEqual(hi[i], 0.); self.assertLess(lo[i], 0.)
        d.qvel[env.v_adr[i]] = 0.
        lo, hi = env._clamp()
        self.assertAlmostEqual(hi[i], 35.)

    def test_actuator_forces_stay_within_the_spec_during_rollouts(self):
        from run_env import RunEnv
        env = RunEnv(randomize=False, seed=1)
        env.reset()
        rng = np.random.default_rng(1)
        over, n = 0, 0
        for _ in range(60):
            _, _, term, _, _ = env.step(rng.uniform(-1, 1, 12))
            tau, qd = env.data.actuator_force[env.act], env.data.qvel[env.v_adr]
            self.assertTrue(np.all(np.abs(tau) <= env.tau_max+1e-6))
            # the power clamp uses the velocity at the start of each 1 ms step; an impact inside the step can
            # raise the ex-post product (measured: 3 of 720 samples, worst 1.7x) - it must stay rare
            over += int(np.sum(tau*qd > 1.2*env.p_max+1.)); n += len(tau)
            if term:
                env.reset()
        self.assertLess(over/n, .02)

    def test_tail_lock_removes_the_tail_from_the_policy(self):
        from run_env import RunEnv
        env = RunEnv(randomize=False, seed=0, tail='locked')
        obs, _ = env.reset()
        self.assertEqual(env.action_space.shape, (10,))
        self.assertEqual(obs.shape, env.observation_space.shape)
        for _ in range(10):
            env.step(np.ones(10))
        tail_q = env.data.qpos[env.q_adr[env.tail_idx]]
        np.testing.assert_allclose(tail_q, env.q0[env.tail_idx], atol=.05)

    def test_balanced_start_and_body_contact_termination(self):
        env = self.env
        env.reset()
        for _ in range(50):  # the passive hold topples after ~2 s; the first second must be clean
            _, _, term, _, _ = env.step(np.zeros(12))
            self.assertFalse(term)
        env.reset()
        env.data.qpos[2] -= .5
        mujoco.mj_forward(env.model, env.data)
        _, reward, term, _, _ = env.step(np.zeros(12))
        self.assertTrue(term); self.assertLess(reward, 0)

    def test_angular_momentum_parts(self):
        env, m, d = self.env, self.env.model, self.env.data
        env.reset()
        d.qvel[:] = 0.
        d.qvel[env.v_adr[env.active.index('tail_yaw_joint')]] = 5.
        mujoco.mj_forward(m, d)
        L, L_tail, L_legs = env.angular_momentum()
        self.assertGreater(abs(L_tail[2]), .01)
        np.testing.assert_allclose(L, L_tail, atol=1e-6)
        np.testing.assert_allclose(L_legs, 0., atol=1e-6)

    def test_yaw_impulse_spins_the_body(self):
        env = self.env
        env.reset()
        env.apply_impulse('z', .5)
        env.step(np.zeros(12)); env.step(np.zeros(12))
        self.assertGreater(abs(env.body_velocity(env.base)[0][2]), .5)


if __name__ == '__main__':
    unittest.main()
