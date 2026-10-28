"""Realism layer (sim/rl/realism.py): thermal steady state, torque-speed limits, exact delays, derate/cut-off, spec
loading, and that realism=None leaves RunEnv untouched."""
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sim'/'rl'))
try:
    import gymnasium  # noqa: F401
    import mujoco
    import numpy as np
except ImportError:  # .venv-ai has no simulator stack; use .venv-sim
    mujoco = None

if mujoco is not None:
    import realism as rz

PLACEHOLDER = ROOT/'sim'/'rl'/'motor_specs'/'PLACEHOLDER_not_real.json'


def only(*blocks, **extra):
    """Realism config with exactly these blocks on."""
    return {'enable': list(blocks), **extra}


@unittest.skipIf(mujoco is None, 'mujoco/gymnasium not installed')
class BuildingBlockTest(unittest.TestCase):
    def test_thermal_steady_state_matches_p_times_r_sum(self):
        th = rz.Thermal2RC([.6, 1.2], [1.5, .8], [2., 5.], [10., 20.])  # small C: fast settling
        th.reset(25.)
        p = np.array([30., 12.])
        for _ in range(40000):
            th.step(p, .01, 25.)
        tw, thh = th.steady(p, 25.)
        np.testing.assert_allclose(th.t_w, 25.+p*(np.array([.6, 1.2])+np.array([1.5, .8])), rtol=1e-6)
        np.testing.assert_allclose(th.t_w, tw, rtol=1e-6)
        np.testing.assert_allclose(th.t_h, thh, rtol=1e-6)

    def test_thermal_fixed_point_with_copper_temperature_coefficient(self):
        """Constant current, R(T) = R0 (1 + a (T - Tref)): T = (Ta + I^2 R0 Rs (1 - a Tref)) / (1 - I^2 R0 a Rs)."""
        I, R0, a, tref, ta, rs = 10., .15, .00393, 25., 30., 2.1
        th = rz.Thermal2RC([.6], [1.5], [2.], [8.])
        th.reset(ta)
        for _ in range(60000):
            th.step(I*I*R0*(1+a*(th.t_w-tref)), .01, ta)
        expect = (ta+I*I*R0*rs*(1-a*tref))/(1-I*I*R0*a*rs)
        self.assertAlmostEqual(float(th.t_w[0]), expect, places=5)

    def test_delay_line_shifts_by_exactly_k(self):
        for k in (0, 1, 3, 7):
            line = rz.DelayLine(k, -1)
            out = [line.push(i) for i in range(20)]
            self.assertEqual(out, [-1]*k+list(range(20-k)))
        with self.assertRaises(ValueError):
            rz.DelayLine(-1, 0)

    def test_derate_is_linear_between_start_and_max(self):
        np.testing.assert_allclose(rz.derate(np.array([50., 100., 110., 120., 130.]), 100., 120.), [1., 1., .5, 0., 0.])

    def test_spec_loader_shapes_and_errors(self):
        specs = rz.load_motor_specs(PLACEHOLDER)
        s = specs['PLACEHOLDER_NOT_REAL_9to1']
        self.assertTrue(s['placeholder'])
        self.assertIn('PLACEHOLDER', PLACEHOLDER.name)
        base = {k: v for k, v in s.items() if k in rz.MOTOR_FIELDS}
        # {name: spec}, value objects with sources, Kelvin
        alt = rz.load_motor_specs({'motors': {'m1': {**base, 'kt': {'value': .2, 'source': 'x'},
                                                     't_winding_max': 393.15, 'temp_unit': 'K'}}})
        self.assertAlmostEqual(alt['m1']['kt'], .2)
        self.assertAlmostEqual(alt['m1']['t_winding_max'], 120.)
        self.assertEqual(alt['m1']['backdrive_efficiency'], base['gear_efficiency'])
        self.assertEqual(set(rz.load_motor_specs([{**base, 'name': 'a'}, {**base, 'name': 'b'}])), {'a', 'b'})
        with self.assertRaisesRegex(ValueError, 'kt'):
            rz.load_motor_specs({**{k: v for k, v in base.items() if k != 'kt'}, 'name': 'bad'})
        with self.assertRaisesRegex(ValueError, 'r_phase'):
            rz.load_motor_specs({**base, 'r_phase': -1.})

    def test_config_merge_and_enable(self):
        self.assertIsNone(rz.load_config(None))
        self.assertIsNone(rz.load_config(False))
        cfg = rz.load_config(only('delay', delay={'action_steps': {'nominal': 3}}))
        self.assertEqual([b for b in rz.BLOCKS if cfg[b]['enabled']], ['delay'])
        self.assertEqual(cfg['delay']['action_steps'], {'nominal': 3, 'range': [0, 6]})
        default = json.loads(rz.DEFAULT_CONFIG.read_text())
        for b in rz.BLOCKS:
            self.assertEqual(default[b]['source'], '')
            self.assertIn('sim2real_params.json', default[b]['source_todo'])
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(only('imu'), f)
        self.assertTrue(rz.load_config(f.name)['imu']['enabled'])
        with self.assertRaises(ValueError):
            rz.load_config({'enable': ['nope']})


@unittest.skipIf(mujoco is None, 'mujoco/gymnasium not installed')
class RealismEnvTest(unittest.TestCase):
    def make(self, cfg, **kw):
        from run_env import RunEnv
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            env = RunEnv(seed=0, realism=cfg, **{'randomize': False, **kw})
        env.reset(seed=0)
        return env

    def test_off_leaves_env_untouched(self):
        from run_env import RunEnv
        a, b = RunEnv(seed=3), RunEnv(seed=3, realism=None)
        self.assertIsNone(b.rz)
        oa, ob = a.reset(seed=3)[0], b.reset(seed=3)[0]
        self.assertEqual(oa.tobytes(), ob.tobytes())
        act = np.full(a.action_space.shape, .3)
        ra, rb = a.step(act), b.step(act)
        self.assertEqual(ra[0].tobytes(), rb[0].tobytes())
        self.assertEqual(ra[1], rb[1])
        self.assertNotIn('realism', rb[4])

    def test_torque_speed_limit(self):
        env = self.make(only('motor'))
        z, d = env.rz, env.data
        i = 0
        kt, R, N, eta, etab = z.kt[i], z.r0[i], z.ratio(d.qpos[z.q_adr_m])[i], z.eta[i], z.eta_b[i]
        V, ipk = z.v_nom[i], z.i_peak[i]
        w0 = V/(kt*N)  # no-load joint speed
        for qd in (0., .2*w0, .5*w0, .9*w0, 1.2*w0, -.5*w0):
            d.qvel[env.v_adr[z.idx[i]]] = qd
            lo, hi = env._clamp()
            wm = N*qd
            drive_hi = min(ipk, max(0., (V-kt*wm)/R))*kt*N*(eta if qd >= 0 else 1/etab)
            self.assertAlmostEqual(hi[z.idx[i]], drive_hi, places=9, msg=qd)
            self.assertAlmostEqual(-lo[z.idx[i]], min(ipk, max(0., (V+kt*wm)/R))*kt*N*(eta if qd <= 0 else 1/etab), places=9)
            self.assertEqual(env.model.actuator_forcerange[env.act[z.idx[i]], 1], hi[z.idx[i]])
            if qd > 0:  # drive quadrant: the back-EMF line tau <= (V - kt w_m)/R kt N eta
                self.assertLessEqual(hi[z.idx[i]], (V-kt*wm)/R*kt*N*eta+1e-9 if wm < V/kt else 1e-12)
        d.qvel[:] = 0.

    def test_rollout_forces_stay_within_motor_limits(self):
        env = self.make(only('motor', 'thermal', 'friction'))
        z, d = env.rz, env.data
        rng = np.random.default_rng(0)
        for _ in range(40):
            lo = env.model.actuator_forcerange[env.act, 0].copy()
            hi = env.model.actuator_forcerange[env.act, 1].copy()
            _, _, term, _, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
            f = d.actuator_force[env.act]
            # forces of the last physics step lie in the range _clamp set for that step
            self.assertTrue(np.all(f >= env.model.actuator_forcerange[env.act, 0]-1e-9))
            self.assertTrue(np.all(f <= env.model.actuator_forcerange[env.act, 1]+1e-9))
            self.assertTrue(np.all(np.isfinite(info['realism']['t_winding'][z.idx])))
            if term:
                break
        peak = z.i_peak*z.kt*z.ratio(d.qpos[z.q_adr_m])*np.maximum(z.eta, 1/z.eta_b)
        self.assertTrue(np.all(np.abs(d.actuator_force[env.act[z.idx]]) <= peak+1e-9))

    def test_armature_and_friction_are_set(self):
        env = self.make(only('motor', 'friction'))
        z, m = env.rz, env.model
        jr = z.nom['j_rotor']*(z.nom['gear_ratio']*z.ext)**2
        np.testing.assert_allclose(m.dof_armature[z.armature_dofs], jr)
        np.testing.assert_allclose(m.dof_frictionloss[z.fric_dofs], z.floss_base+z.coulomb)
        np.testing.assert_allclose(m.dof_damping[z.fric_dofs], z.damp_base+z.viscous)

    def test_derate_and_cutoff(self):
        env = self.make(only('motor', 'thermal'))
        z, d = env.rz, env.data
        d.qvel[:] = 0.
        lo0, hi0 = env._clamp()
        i = z.idx[0]
        z.thermal.t_w[0] = z.t_derate[0]+.25*(z.t_max[0]-z.t_derate[0])  # a quarter into the derating band
        z._thermal_state()
        r_scale = (1+z.alpha*(z.thermal.t_w[0]-z.t_ref))/(1+z.alpha*(z.t_amb-z.t_ref))
        lo, hi = env._clamp()
        # at zero speed the current limit binds (V/R >> i_peak), so the torque scales by the derate factor alone
        self.assertGreater(z.v_nom[0]/(z.r0[0]*r_scale), z.i_peak[0])
        self.assertAlmostEqual(hi[i], .75*hi0[i], places=9)
        z.thermal.t_w[0] = z.t_max[0]+1.
        env.rz.after_substep(env, 0.)  # dt 0: only the cut-off latch updates
        self.assertTrue(z.cut[0])
        lo, hi = env._clamp()
        self.assertEqual((lo[i], hi[i]), (0., 0.))
        z.thermal.t_w[0] = z.t_derate[0]+1.  # cooled into the band: still latched
        env.rz.after_substep(env, 0.)
        self.assertEqual(env._clamp()[1][i], 0.)
        z.thermal.t_w[0] = z.t_derate[0]-1.  # below the derate start: released, full torque
        env.rz.after_substep(env, 0.)
        self.assertFalse(z.cut[0])
        self.assertGreater(env._clamp()[1][i], .99*hi0[i])

    def test_action_delay_shifts_servo_targets_by_k_physics_steps(self):
        rec = {}
        for k in (0, 7, 23):
            env = self.make(only('delay', delay={'action_steps': {'nominal': k}, 'obs_steps': {'nominal': 0}}))
            seen = []
            orig = env._clamp
            env._clamp = lambda: (seen.append(env.data.ctrl[env.act].copy()), orig())[1]
            rng = np.random.default_rng(5)
            for _ in range(4):
                env.step(rng.uniform(-1, 1, env.action_space.shape))
            rec[k] = np.array(seen)
        # targets do not depend on the state (no ramp), so the applied sequence is the undelayed one shifted by k
        for k in (7, 23):
            np.testing.assert_array_equal(rec[k][k:], rec[0][:-k])
            np.testing.assert_array_equal(rec[k][:k], np.repeat(rec[k][:1], k, 0))

    def test_observation_delay_shifts_sensors_by_k_physics_steps(self):
        n_sub = 20
        obs = {}
        for k in (0, n_sub, 2*n_sub):
            env = self.make(only('delay', delay={'action_steps': {'nominal': 0}, 'obs_steps': {'nominal': k}}))
            o0 = env._observe()
            rng = np.random.default_rng(6)
            obs[k] = [o0]+[env.step(rng.uniform(-1, 1, env.action_space.shape))[0] for _ in range(5)]
            jq = slice(11+3, 11+3+len(env.active))  # joint-position part (obs_vel on)
        for k, lag in ((n_sub, 1), (2*n_sub, 2)):
            for t in range(lag, 6):
                np.testing.assert_array_equal(obs[k][t][jq], obs[0][t-lag][jq])
            np.testing.assert_array_equal(obs[k][1][jq], obs[0][0][jq])  # before that: the reset state

    def test_backlash_dead_band(self):
        env = self.make(only('backlash'))
        b = env.rz.backlash
        q = env.data.qpos[env.q_adr].copy()
        want = q+np.r_[.3*b[:6], -2*b[6:]]
        out = env.rz.command(env, want)
        np.testing.assert_allclose(out[:6], q[:6])                    # inside the gap: no servo error
        np.testing.assert_allclose(out[6:], q[6:]-1.5*b[6:])          # outside: error reduced by half the gap

    def test_sensor_noise_and_thermal_observation(self):
        env = self.make(only('motor', 'thermal', 'imu', 'encoder', thermal={'observe': True}))
        n = len(env.active)
        self.assertEqual(env.observation_space.shape[0], env._obs().shape[0]+n)
        seen = []
        compose = env._compose
        env._compose = lambda s: (seen.append(s), compose(s))[1]
        o = env._observe()
        self.assertEqual(o.shape, env.observation_space.shape)
        clean = env._sense()
        self.assertFalse(np.array_equal(seen[0]['gyro'], clean['gyro']))  # gyro noise
        res = 2*np.pi/2**env.rz.cfg['encoder']['bits']
        q = seen[0]['q']/res
        np.testing.assert_allclose(q, np.round(q), atol=1e-9)   # quantised encoder
        self.assertLess(np.abs(seen[0]['q']-clean['q']).max(), .01)
        np.testing.assert_allclose(o[-n:], 0., atol=1e-9)        # at ambient

    def test_battery_sag(self):
        env = self.make(only('motor', 'battery'))
        z = env.rz
        env.step(np.ones(env.action_space.shape))
        self.assertGreater(z.i_bus, 0.)
        self.assertAlmostEqual(z.v_bus, max(z.v_oc-z.r_int*z.i_bus, .1*z.v_oc), places=9)
        self.assertLess(z.v_bus, z.v_oc)

    def test_randomized_parameters_stay_in_range_and_are_seeded(self):
        from run_env import RunEnv
        cfg = rz.load_config(True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a, b = RunEnv(seed=11, realism=True), RunEnv(seed=11, realism=True)
        a.reset(seed=11); b.reset(seed=11)
        for z in (a.rz,):
            lo, hi = cfg['delay']['action_steps']['range']
            self.assertTrue(lo <= z.k_act <= hi)
            self.assertTrue(10. <= z.t_amb <= 40.)
            f = z.kt/z.nom['kt']
            self.assertTrue(np.all((f >= .95) & (f <= 1.05)))
        np.testing.assert_array_equal(a.rz.kt, b.rz.kt)
        act = np.zeros(a.action_space.shape)
        self.assertEqual(a.step(act)[0].tobytes(), b.step(act)[0].tobytes())


if __name__ == '__main__':
    unittest.main()
