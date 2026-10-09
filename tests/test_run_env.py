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

    def test_ankle_clutch_lets_the_swing_foot_fold(self):
        """Unclutched, the Achilles spring stops an airborne ankle near 88 deg (motor 22 N·m); clutched it folds."""
        from run_env import RunEnv
        reached = {}
        for clutch in (False, True):
            env = RunEnv(randomize=False, seed=0, ankle_clutch=clutch)
            env.reset()
            env.data.qpos[2] += 1.
            mujoco.mj_forward(env.model, env.data)
            env.model.opt.gravity[:] = 0
            env.loaded = {'left': False, 'right': False}
            a = np.zeros(12); a[env.ix['left_ankle_pitch_joint']] = -1.
            for _ in range(25):
                env.step(a)
            reached[clutch] = 180+np.degrees(env.data.qpos[env.q_adr[env.ix['left_ankle_pitch_joint']]])
        self.assertGreater(reached[False], 80.)
        self.assertLess(reached[True], 50.)

    def test_yaw_impulse_spins_the_body(self):
        env = self.env
        env.reset()
        env.apply_impulse('z', .5)
        env.step(np.zeros(12)); env.step(np.zeros(12))
        self.assertGreater(abs(env.body_velocity(env.base)[0][2]), .5)

    def test_structure_variants_contract(self):
        """Equal-condition structure comparison (2026-10-01): arch8 / arch12 with explicit part masses and the same motor
        classes, and the R-03 hip mount moved rearward with the torso payload and tail root fixed."""
        from run_env import MOTOR_CLASSES, LINK_ETA, RunEnv
        e8 = RunEnv(arch8=True, real_mass=True, randomize=False, seed=0); e8.reset()
        e12 = RunEnv(arch12=True, real_mass=True, randomize=False, seed=0); e12.reset()
        self.assertEqual(e8.action_space.shape, (8,))
        self.assertEqual(e12.action_space.shape, (12,))
        self.assertAlmostEqual(e8.model.body_subtreemass[e8.root], 11.29, delta=.02)
        self.assertAlmostEqual(e12.model.body_subtreemass[e12.root], 14.45, delta=.02)
        self.assertAlmostEqual(e12.mass, float(e12.model.body_mass.sum()), delta=1e-6)
        # 12-axis ankle pitch: class B through 1.5:1 at rest -> 48 x 1.5 x 0.9 = 64.8 N·m, and its torque heats a motor
        i = e12.active.index('left_ankle_pitch_joint')
        lo, hi = e12._clamp()
        self.assertAlmostEqual(hi[i], MOTOR_CLASSES['B'][0]*1.5*LINK_ETA, delta=.5)
        self.assertLess(e12.motor_cont[i], 1e8)
        self.assertGreater(e8.motor_cont[e8.active.index('left_ankle_pitch_joint')], 1e8)  # coupled, no motor
        for _ in range(10):
            e12.step(np.zeros(12))
        self.assertGreater(e12.heat[i], 0.)
        # R-03: hips 0.2 m further back on the torso, tail root unchanged
        eb = RunEnv(arch8=True, real_mass=True, hip_back=.2, randomize=False, seed=0); eb.reset()
        hip = lambda env: env.model.body_pos[env.model.body('left_hip_roll_link').id][0]
        tail = lambda env: env.model.body_pos[env.model.body('tail_yaw_link').id][0]
        self.assertAlmostEqual(hip(eb)-hip(e8), -.2, places=6)
        self.assertAlmostEqual(tail(eb), tail(e8), places=6)

    def test_evaluation_starts_differ_by_init_seed(self):
        """Deterministic evaluation episodes with different init_seed start from different states; None keeps the
        nominal start (before 2026-10-01 every evaluation episode started identically)."""
        from run_env import RunEnv
        def start(seed):
            env = RunEnv(randomize=False, seed=0); env.init_seed = seed; env.reset()
            return env.data.qpos.copy(), env.phase
        (a, pa), (b, pb), (c, _), (d, _) = start(1), start(2), start(None), start(None)
        self.assertFalse(np.allclose(a, b))
        self.assertNotEqual(pa, pb)
        self.assertTrue(np.allclose(c, d))


if __name__ == '__main__':
    unittest.main()
