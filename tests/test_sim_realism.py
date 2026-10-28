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
SURVEY = ROOT/'docs'/'design'/'motor_candidates.json'
ROBSTRIDE = ROOT/'sim'/'rl'/'realism_robstride.json'


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

    def test_survey_modules_load_with_estimated_thermal_overrides(self):
        over = json.loads(ROBSTRIDE.read_text())['motor']['spec_override']
        specs = rz.load_motor_specs(SURVEY, over, {'RS03', 'RS06'})
        self.assertEqual(set(specs), {'RS03', 'RS06'})  # other vendors (incomplete data) are not parsed
        mods = json.loads(SURVEY.read_text())['modules']
        for name, s in specs.items():
            m = mods[name]
            self.assertFalse(s['placeholder'])
            self.assertAlmostEqual(s['kt']*s['gear_ratio'], m['kt_out'])
            self.assertAlmostEqual(s['i_peak']*s['kt']*s['gear_ratio'], m['tau_peak'])   # published peak torque
            self.assertAlmostEqual(s['i_cont']*s['kt']*s['gear_ratio'], m['tau_cont'])
            self.assertAlmostEqual(s['j_rotor']*s['gear_ratio']**2, m['j_out'])
            self.assertEqual((s['r_phase'], s['t_winding_max'], s['gear_efficiency']), (m['r_line'], m['t_wind'], 1.))
            self.assertIn('RobStride', s['source'])
            for f in ('r_th_wh', 'r_th_ha', 'c_th_w', 'c_th_h'):
                self.assertTrue(over[name][f]['estimate'])
                self.assertIn('maxon', over[name][f]['source'])
                self.assertEqual(s[f], over[name][f]['value'])
        with self.assertRaisesRegex(ValueError, 'r_th_wh'):  # the survey has no thermal data of its own
            rz.load_motor_specs(SURVEY, names={'RS03'})

    def test_config_merge_and_enable(self):
        self.assertIsNone(rz.load_config(None))
        self.assertIsNone(rz.load_config(False))
        cfg = rz.load_config(only('delay', delay={'action_steps': {'nominal': 3}}))
        self.assertEqual([b for b in rz.BLOCKS if cfg[b]['enabled']], ['delay'])
        self.assertEqual(cfg['delay']['action_steps'], {'nominal': 3, 'range': [0, 6]})
        default = json.loads(rz.DEFAULT_CONFIG.read_text())
        params = json.loads((ROOT/'docs'/'design'/'sim2real_params.json').read_text())['params']
        names = {p['name'] for p in params}
        for b in rz.BLOCKS:  # every source names entries that exist in sim2real_params.json
            src = default[b]['source']
            self.assertEqual(src['file'], 'docs/design/sim2real_params.json')
            cited = {w.strip(',;:()') for v in src.values() for w in v.split()} & names
            self.assertTrue(cited, b)
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
        tq = lambda cur, qd: cur*kt*N*(eta if cur*qd >= 0 else 1/etab)
        for qd in (0., .2*w0, .5*w0, .9*w0, 1.1*w0, 1.2*w0, 2.*w0, -.5*w0, -1.1*w0):
            d.qvel[env.v_adr[z.idx[i]]] = qd
            lo, hi = env._clamp()
            E = kt*N*qd
            c_hi, c_lo = min(ipk, (V-E)/R), max(-ipk, (-V-E)/R)  # current limit intersected with the voltage window
            if c_lo > c_hi:
                c_lo = c_hi = -np.sign(E)*ipk
            self.assertAlmostEqual(hi[z.idx[i]], tq(c_hi, qd), places=9, msg=qd)
            self.assertAlmostEqual(lo[z.idx[i]], tq(c_lo, qd), places=9, msg=qd)
            self.assertLessEqual(lo[z.idx[i]], hi[z.idx[i]])
            self.assertEqual(env.model.actuator_forcerange[env.act[z.idx[i]], 1], hi[z.idx[i]])
            if 0 < qd < w0:  # drive quadrant: the back-EMF line tau <= (V - kt w_m)/R kt N eta
                self.assertLessEqual(hi[z.idx[i]], (V-E)/R*kt*N*eta+1e-9)
            if qd > w0:  # above no-load speed only braking (negative) torque is possible
                self.assertLess(hi[z.idx[i]], 0.)
        d.qvel[:] = 0.

    def test_current_window_overspeed(self):
        """E 60 V > V 48 V, R 1 ohm, 20 A: physically -20 .. -12 A (the earlier code allowed -20 .. 0 A)."""
        lo, hi, empty = rz.current_window(np.array([60., -60., 10., 0.]), 48., 1., 20.)
        np.testing.assert_allclose(lo, [-20., 12., -20., -20.])
        np.testing.assert_allclose(hi, [-12., 20., 20., 20.])
        self.assertFalse(empty.any())
        # |E| > V + i_lim R: no overlap -> forced regenerative current at the limit; cut-off (i_lim 0) -> 0
        lo, hi, empty = rz.current_window(np.array([80., -80., 80.]), 48., 1., np.array([20., 20., 0.]))
        np.testing.assert_array_equal(empty, [True, True, True])
        np.testing.assert_allclose(lo, [-20., 20., 0.])
        np.testing.assert_allclose(hi, [-20., 20., 0.])

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

    def backlash_env(self, b=.04, **kw):
        env = self.make(only('backlash', backlash={'rad': {k: b for k in rz.json.loads(rz.DEFAULT_CONFIG.read_text())['backlash']['rad']},
                                                   'dr': {}}), **kw)
        return env, env.rz

    def test_backlash_free_play_no_joint_torque_inside_gap(self):
        env, z = self.backlash_env()
        d, m = env.data, env.model
        j = 0
        z.bl_s[:] = 0.
        z.bl_th[:], z.bl_w[:] = d.qpos[z.bl_q], d.qvel[z.bl_dof]
        d.ctrl[env.act] = d.qpos[env.q_adr]
        d.ctrl[z.bl_act[j]] += .5  # servo pulls hard in +
        lo, hi = env._clamp()
        th0, w0 = z.bl_th[j], z.bl_w[j]
        z.before_substep(env, lo, hi, 1e-3)
        self.assertEqual(tuple(m.actuator_forcerange[z.bl_act[j]]), (0., 0.))  # joint feels nothing in the gap
        self.assertEqual(m.dof_armature[z.bl_dof[j]], 0.)                       # rotor not on the joint
        self.assertGreater(z.bl_w[j], w0)                                         # rotor accelerates in +
        self.assertGreater(z.bl_th[j], th0)
        self.assertEqual(z.bl_s[j], 0.)
        tau = z.bl_tau[j]
        self.assertGreater(tau, 0.)
        self.assertAlmostEqual(z.bl_w[j], w0+1e-3*tau/z.bl_J[j], places=9)

    def test_backlash_recontact_impulse_and_engagement(self):
        env, z = self.backlash_env()
        d, m = env.data, env.model
        j = len(z.bl_idx)-1  # tail joint: no ground contact on that branch
        q = d.qpos[z.bl_q[j]]
        h = .5*z.backlash[j]
        d.qvel[:] = 0.
        mujoco.mj_forward(m, d)
        z.bl_s[:] = 0.
        z.bl_th[:], z.bl_w[:] = d.qpos[z.bl_q], 0.
        z.bl_J[j] = .02  # an RS03-like rotor (output side)
        z.bl_th[j], z.bl_w[j] = q+h-1e-4, 3.  # rotor about to hit the + face at 3 rad/s
        d.ctrl[env.act] = d.qpos[env.q_adr]
        d.ctrl[z.bl_act[j]] = z.bl_th[j]
        lo, hi = env._clamp()
        n0 = z.bl_impacts
        z.before_substep(env, lo, hi, 1e-3)
        self.assertEqual(z.bl_impacts, n0+1)
        self.assertEqual(z.bl_s[j], 1.)  # plastic: engaged on the + face
        tau = m.actuator_forcerange[z.bl_act[j], 0]
        self.assertEqual(tau, m.actuator_forcerange[z.bl_act[j], 1])
        x = np.zeros((1, m.nv)); e = np.zeros((1, m.nv)); e[0, z.bl_dof[j]] = 1.
        mujoco.mj_solveM(m, d, x, e)
        jl, J = 1/x[0, z.bl_dof[j]], z.bl_J[j]
        self.assertGreater(tau, 0.)
        # impulse = rotor speed at contact x reduced inertia; the rotor speed after this 1 ms of servo damping
        # (implicit PD, target at the rotor) is w_f = 3 / (1 + dt kd/J + dt^2 kp/J)
        kp, kd = z.bl_kp[j], z.bl_kd[j]
        w_f = 3./(1.+1e-3*kd/J+1e-6*kp/J)
        self.assertAlmostEqual(tau*1e-3, w_f*J*jl/(J+jl), delta=1e-6*w_f*J)
        qd0 = d.qvel[z.bl_dof[j]]
        mujoco.mj_step(m, d)
        self.assertGreater(d.qvel[z.bl_dof[j]]-qd0, .5*tau*1e-3/jl)  # the joint is knocked forward
        lo, hi = env._clamp()
        z.before_substep(env, lo, hi, 1e-3)  # engaged: rotor inertia back on the joint
        if z.bl_s[j] != 0:
            self.assertEqual(m.dof_armature[z.bl_dof[j]], J)

    def test_backlash_contact_opens_when_servo_torque_reverses(self):
        env, z = self.backlash_env()
        d, m = env.data, env.model
        j = 0
        z.bl_s[:] = 1.
        d.ctrl[env.act] = d.qpos[env.q_adr]
        d.ctrl[z.bl_act[j]] -= .5  # reverse: servo now pushes in -
        d.qacc[:] = 0.
        lo, hi = env._clamp()
        z.before_substep(env, lo, hi, 1e-3)
        self.assertEqual(z.bl_s[j], 0.)
        self.assertEqual(tuple(m.actuator_forcerange[z.bl_act[j]]), (0., 0.))
        self.assertLess(z.bl_w[j], d.qvel[z.bl_dof[j]])  # rotor starts back across the gap
        # a joint whose servo keeps pushing into the face stays engaged and transmits the servo torque
        k = 1
        d.ctrl[z.bl_act[k]] = d.qpos[z.bl_q[k]]+1.
        z.bl_s[k] = 1.
        lo, hi = env._clamp()
        z.before_substep(env, lo, hi, 1e-3)
        self.assertEqual(z.bl_s[k], 1.)
        self.assertGreater(m.actuator_forcerange[z.bl_act[k], 0], 0.)

    def test_backlash_rollout_finite_with_impacts(self):
        env, z = self.backlash_env(b=.01)
        rng = np.random.default_rng(3)
        a = np.zeros(env.action_space.shape)
        for _ in range(60):
            a = .7*a+.3*rng.uniform(-1, 1, a.shape)
            obs, _, term, _, info = env.step(a)
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertTrue(np.all(np.isfinite(env.data.qvel)))
            if term:
                break
        self.assertGreater(info['realism']['backlash_impacts'], 0)

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

    def test_robstride_config_on_arch8(self):
        env = self.make(str(ROBSTRIDE), arch8=True, model_rev=2)
        z, d, m = env.rz, env.data, env.model
        names = {env.active[i]: z.motor_name[k] for k, i in enumerate(z.idx)}
        self.assertEqual(names, {'left_hip_pitch_joint': 'RS03', 'right_hip_pitch_joint': 'RS03',
                                 'left_hip_roll_joint': 'RS06', 'right_hip_roll_joint': 'RS06',
                                 'left_knee_pitch_joint': 'RS03', 'right_knee_pitch_joint': 'RS03',
                                 'tail_pitch_joint': 'RS03', 'tail_yaw_joint': 'RS06'})
        d.qvel[:] = 0.
        lo, hi = env._clamp()
        N_fb = np.interp(d.qpos[z.q_adr_m], *env.knee_ratio)
        expect = {'RS03': 60., 'RS06': 36.}
        for k, i in enumerate(z.idx):  # stall torque = published module peak x external ratio x link efficiency
            ext = N_fb[k] if z.fourbar[k] else z.ext[k]
            self.assertAlmostEqual(hi[i], expect[z.motor_name[k]]*ext*z.link[k], places=6, msg=env.active[i])
            self.assertAlmostEqual(lo[i], -hi[i], places=6)
        hp = env.active.index('left_hip_pitch_joint')
        self.assertAlmostEqual(m.dof_armature[env.v_adr[hp]], .02)  # RS03 output-side inertia, direct drive
        self.assertAlmostEqual(z.loss_factor, 1.5)
        for _ in range(5):
            _, _, term, _, info = env.step(np.zeros(env.action_space.shape))
        for key in ('t_winding', 'current'):
            self.assertTrue(np.all(np.isfinite(info['realism'][key][z.idx])))
        self.assertTrue(np.all(info['realism']['t_winding'][z.idx] >= z.t_amb))

    def test_imu_bias_random_walk_follows_sim_time_not_reads(self):
        env = self.make(only('imu'))
        z, d = env.rz, env.data
        b0 = z.gyro_bias.copy()
        for _ in range(5):  # same sim time: no bias change however often the sensors are read
            env._observe()
        np.testing.assert_array_equal(z.gyro_bias, b0)
        rw, dt, k = z.cfg['imu']['gyro_bias_rw'], .02, 3000
        inc = []
        for _ in range(k):
            d.time += dt
            prev = z.gyro_bias.copy()
            env._observe(); env._observe()  # two reads per step
            inc.append(z.gyro_bias-prev)
        inc = np.array(inc)
        self.assertAlmostEqual(float(inc.std()), rw*np.sqrt(dt), delta=.06*rw*np.sqrt(dt))
        # one long interval: same variance as many short ones
        z.gyro_bias[:] = 0.
        big = []
        for _ in range(k):
            z.gyro_bias[:] = 0.
            d.time += 1.
            env._observe()
            big.append(z.gyro_bias.copy())
        self.assertAlmostEqual(float(np.std(big)), rw, delta=.06*rw)

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
