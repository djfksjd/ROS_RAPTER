"""Sim-realism layer for RunEnv (opt-in, 2026-10-29): motor electrical limits, two-node winding thermal model, joint
friction, backlash, action/observation delay, IMU/encoder noise and battery sag. docs/SIM_REALISM.ko.md.

Nothing here runs unless RunEnv(realism=...) is given. With realism=None the env takes exactly its earlier code path
(realism_regression.py compares rollouts bit for bit against main). Every parameter is an ASSUMPTION until the motor
survey (docs/design/motor_candidates.json) and docs/design/sim2real_params.json exist; the default spec file
motor_specs/PLACEHOLDER_not_real.json is not a real motor.

Motor model, per driven joint (spec fields SI, DC-equivalent: one current i = tau_m/kt, one resistance R, back-EMF
constant = kt). Total ratio N = gear_ratio x external_ratio, eta = gear_efficiency x link_efficiency.
  current limit   i <= i_peak x derate(T_winding)            (0 while the over-temperature cut-off is latched)
  voltage limit   positive torque: i <= (V - kt w_m)/R(T), negative: i <= (V + kt w_m)/R(T), w_m = N qd
                  (in the drive quadrant this is the torque-speed line tau <= (V - kt w_m)/R kt N eta; braking gets the
                  back-EMF as extra headroom)
  output torque   drive (tau qd >= 0): tau = i kt N eta; back-driven (braking): tau = i kt N / eta_b
  rotor inertia   j_rotor N^2 -> MuJoCo dof_armature (replaces the model_rev 2 AK10-9 value on those joints)
The PD gains of RunEnv are unchanged; only the force range of each physics step comes from this model.
Thermal: per motor C_w dT_w = (P - (T_w - T_h)/R_wh) dt, C_h dT_h = ((T_w - T_h)/R_wh - (T_h - T_a)/R_ha) dt with
P = loss_factor i^2 R0 (1 + alpha_cu (T_w - t_ref)); torque derated linearly from T_max - derate_margin to 0 at T_max,
cut off at T_max until the winding is back below the derate start. Explicit Euler at the 1 ms physics step (time
constants are seconds). Temperatures are degC.
"""
import copy
import json
from collections import deque
from pathlib import Path
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE/'realism_default.json'
BLOCKS = ('motor', 'friction', 'backlash', 'thermal', 'delay', 'imu', 'encoder', 'battery')
MOTOR_FIELDS = ('gear_ratio', 'gear_efficiency', 'kt', 'r_phase', 'j_rotor', 'v_bus', 'i_peak', 'i_cont',
                't_winding_max', 'r_th_wh', 'r_th_ha', 'c_th_w', 'c_th_h')
G = 9.81


def joint_type(name):
    return name.removesuffix('_joint').removeprefix('left_').removeprefix('right_')


# --- configuration -------------------------------------------------------------------------------
def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(realism):
    """None/False -> None (layer off). True -> realism_default.json. A path or a dict is merged onto the default
    (nested dicts key by key). Optional top-level 'enable': [block, ...] switches exactly those blocks on."""
    if realism is None or realism is False:
        return None
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    base_dir = HERE
    if isinstance(realism, (str, Path)):
        base_dir = Path(realism).resolve().parent
        realism = json.loads(Path(realism).read_text())
    if isinstance(realism, dict):
        cfg = _merge(cfg, realism)
    elif realism is not True:
        raise TypeError(f'realism: expected None, bool, path or dict, got {type(realism).__name__}')
    if 'enable' in cfg:
        unknown = set(cfg['enable'])-set(BLOCKS)
        if unknown:
            raise ValueError(f'realism enable: unknown blocks {sorted(unknown)}')
        for b in BLOCKS:
            cfg[b]['enabled'] = b in cfg['enable']
    spec = Path(cfg['motor']['spec_file'])
    if not spec.is_absolute():
        spec = base_dir/spec if (base_dir/spec).exists() else HERE/spec
    cfg['motor']['spec_file'] = str(spec)
    return cfg


def load_motor_specs(src):
    """name -> spec (floats, temperatures in degC). `src`: path or parsed JSON. Accepted shapes: one spec, a list of
    specs, or {'motors' | 'candidates': list or {name: spec}}. A field may be a number or {'value': number, ...}
    (source/notes are ignored here). 'backdrive_efficiency' is optional (default gear_efficiency); temp_unit 'K' converts
    t_winding_max. Missing or non-positive required fields raise ValueError naming the motor and the fields."""
    name0 = 'motor'
    if isinstance(src, (str, Path)):
        name0 = Path(src).stem
        src = json.loads(Path(src).read_text())
    file_placeholder = isinstance(src, dict) and bool(src.get('placeholder'))
    if isinstance(src, dict) and ('motors' in src or 'candidates' in src):
        items = src.get('motors', src.get('candidates'))
    else:
        items = src
    if isinstance(items, dict) and not set(MOTOR_FIELDS) & set(items):
        items = [{**v, 'name': v.get('name', k)} for k, v in items.items()]
    if isinstance(items, dict):
        items = [items]
    val = lambda v: v.get('value') if isinstance(v, dict) else v
    out = {}
    for i, raw in enumerate(items):
        name = str(raw.get('name', name0 if len(items) == 1 else f'{name0}_{i}'))
        spec, bad = {}, []
        for f in MOTOR_FIELDS+('backdrive_efficiency',):
            v = val(raw.get(f))
            if v is None:
                if f != 'backdrive_efficiency':
                    bad.append(f)
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                bad.append(f); continue
            if not np.isfinite(v) or (v <= 0 and f != 't_winding_max'):
                bad.append(f); continue
            spec[f] = v
        if bad:
            raise ValueError(f'motor spec {name!r}: missing or invalid fields {bad}')
        if spec['gear_efficiency'] > 1 or spec.get('backdrive_efficiency', 0.) > 1:
            raise ValueError(f'motor spec {name!r}: efficiency above 1')
        spec.setdefault('backdrive_efficiency', spec['gear_efficiency'])
        if str(raw.get('temp_unit', 'C')).upper() == 'K':
            spec['t_winding_max'] -= 273.15
        spec['placeholder'] = file_placeholder or bool(raw.get('placeholder'))
        out[name] = spec
    return out


# --- building blocks -----------------------------------------------------------------------------
class DelayLine:
    """Delay of exactly k pushes: push(x) returns the value pushed k calls earlier (`init` before that); k = 0 passes
    x through."""

    def __init__(self, k, init):
        self.k = int(k)
        if self.k < 0:
            raise ValueError('delay must be >= 0')
        self.buf = deque([init]*(self.k+1), maxlen=self.k+1)

    def push(self, x):
        self.buf.append(x)
        return self.buf[0]


def derate(t, t_derate, t_max):
    """Current-limit factor: 1 up to t_derate, linear to 0 at t_max."""
    return np.minimum(np.maximum((t_max-t)/np.maximum(t_max-t_derate, 1e-9), 0.), 1.)


class Thermal2RC:
    """Winding-housing-ambient two-node RC network per motor (vectorised), explicit Euler."""

    def __init__(self, r_wh, r_ha, c_w, c_h):
        self.r_wh, self.r_ha, self.c_w, self.c_h = (np.asarray(x, float) for x in (r_wh, r_ha, c_w, c_h))
        self.t_w = self.t_h = None

    def reset(self, t_w, t_h=None):
        self.t_w = np.array(np.broadcast_to(t_w, self.r_wh.shape), float)
        self.t_h = self.t_w.copy() if t_h is None else np.array(np.broadcast_to(t_h, self.r_wh.shape), float)

    def step(self, p, dt, t_amb):
        q_wh = (self.t_w-self.t_h)/self.r_wh
        q_ha = (self.t_h-t_amb)/self.r_ha
        self.t_w = self.t_w+dt*(p-q_wh)/self.c_w
        self.t_h = self.t_h+dt*(q_wh-q_ha)/self.c_h

    def steady(self, p, t_amb):
        """Analytic steady state for a constant loss p: (T_w, T_h)."""
        return t_amb+p*(self.r_wh+self.r_ha), t_amb+p*self.r_ha


# --- the layer -----------------------------------------------------------------------------------
class Realism:
    """Holds the sampled plant/sensor parameters and the per-episode state. RunEnv calls setup() at the end of _build,
    reset() at the end of reset, command()/after_substep() in the physics loop, motor_limits() from _clamp and
    observe() from _observe."""

    def __init__(self, cfg, seed=None, control_dt=.02):
        self.cfg, self.control_dt = cfg, float(control_dt)
        self.on = {b: bool(cfg[b].get('enabled')) for b in BLOCKS}
        for b in ('thermal', 'battery'):
            if self.on[b] and not self.on['motor']:
                raise ValueError(f'realism: {b} needs the motor block')
        self.reseed(seed)
        self.specs = load_motor_specs(cfg['motor']['spec_file']) if self.on['motor'] else {}
        used = {a['motor'] for a in cfg['motor']['assign'].values()} if self.on['motor'] else set()
        missing = used-set(self.specs)
        if missing:
            raise ValueError(f'realism motor assign: {sorted(missing)} not in {cfg["motor"]["spec_file"]}')
        if any(self.specs[u]['placeholder'] for u in used):
            warnings.warn('realism: PLACEHOLDER motor spec in use - not a real motor, results are not reportable', stacklevel=3)
        self.obs_temp = self.on['thermal'] and bool(cfg['thermal'].get('observe'))
        self.k_act = self.k_obs = 0

    def reseed(self, seed):
        # own stream: the env's rng draws stay the same whether the layer is on or not
        self.rng = np.random.default_rng(None if seed is None else [int(seed), 0x5EA1])

    def _draw(self, r, nominal=1., size=None):
        if r is None or not self.randomize:
            return nominal if size is None else np.full(size, nominal, float)
        return self.rng.uniform(r[0], r[1], size)

    # --- model ---------------------------------------------------------------------------------
    def setup(self, env, m):
        self.n = len(env.active)
        jt = [joint_type(a) for a in env.active]
        self.v_adr = env.v_adr
        if self.on['motor']:
            assign = self.cfg['motor']['assign']
            self.idx = np.array([i for i in range(self.n) if jt[i] in assign and env.tau_nom[i] > 0], int)
            sp = [self.specs[assign[jt[i]]['motor']] for i in self.idx]
            a = [assign[jt[i]] for i in self.idx]
            self.motor_name = [assign[jt[i]]['motor'] for i in self.idx]
            self.nom = {f: np.array([s[f] for s in sp]) for f in MOTOR_FIELDS+('backdrive_efficiency',)}
            link = np.array([float(x.get('link_efficiency', 1.)) for x in a])
            self.fourbar = np.array([x.get('external_ratio', 1.) == 'fourbar' for x in a])
            if self.fourbar.any() and not hasattr(env, 'knee_ratio'):
                raise ValueError("external_ratio 'fourbar' needs arch8")
            self.ext = np.array([1. if fb else float(x.get('external_ratio', 1.)) for fb, x in zip(self.fourbar, a)])
            self.link = link
            if self.fourbar.any():
                qq, N = env.knee_ratio
                lo_, hi_ = env.knee_band if env.knee_band is not None else (qq[0], qq[-1])
                band = (qq >= lo_) & (qq <= hi_)
                self.ext2 = np.where(self.fourbar, float(np.mean(N[band]**2)), self.ext**2)  # armature: mean N^2 (rev 2 rule)
            else:
                self.ext2 = self.ext**2
            self.knee_ratio = getattr(env, 'knee_ratio', None)
            self.q_adr_m, self.v_adr_m, self.act_m = env.q_adr[self.idx], env.v_adr[self.idx], env.act[self.idx]
            self.has_fourbar = bool(self.fourbar.any())
            self.n_const = self.nom['gear_ratio']*self.ext
            self.armature_dofs = env.v_adr[self.idx]
            self.armature_base = m.dof_armature[self.armature_dofs].copy()
        else:
            self.idx = np.zeros(0, int)
        if self.on['friction']:
            fc = self.cfg['friction']
            sel = [i for i in range(self.n) if jt[i] in fc['coulomb'] or jt[i] in fc['viscous']]
            self.fric_dofs = env.v_adr[sel]
            self.coulomb = np.array([float(fc['coulomb'].get(jt[i], 0.)) for i in sel])
            self.viscous = np.array([float(fc['viscous'].get(jt[i], 0.)) for i in sel])
            self.floss_base = m.dof_frictionloss[self.fric_dofs].copy()
            self.damp_base = m.dof_damping[self.fric_dofs].copy()
        if self.on['backlash']:
            self.backlash_nom = np.array([float(self.cfg['backlash']['rad'].get(t, 0.)) for t in jt])

    def ratio(self, q_m):
        """Total motor-to-joint ratio of the motor joints at joint angles q_m (four-bar: at the current knee angle)."""
        if not self.has_fourbar:
            return self.n_const
        N = self.n_const.copy()
        N[self.fourbar] = self.nom['gear_ratio'][self.fourbar]*np.interp(q_m[self.fourbar], *self.knee_ratio)
        return N

    # --- episode -------------------------------------------------------------------------------
    def reset(self, env):
        m, d = env.model, env.data
        self.randomize = bool(env.randomize)
        if self.on['motor']:
            dr = self.cfg['motor'].get('dr', {})
            k = len(self.idx)
            self.kt = self.nom['kt']*self._draw(dr.get('kt'), size=k)
            self.r0 = self.nom['r_phase']*self._draw(dr.get('r_phase'), size=k)
            f_eta = self._draw(dr.get('gear_efficiency'), size=k)  # one factor for both directions
            self.eta = np.minimum(self.nom['gear_efficiency']*f_eta, 1.)*self.link
            self.eta_b = np.minimum(self.nom['backdrive_efficiency']*f_eta, 1.)*self.link
            self.i_peak = self.nom['i_peak']*self._draw(dr.get('i_peak'), size=k)
            jr = self.nom['j_rotor']*self._draw(dr.get('j_rotor'), size=k)
            if self.cfg['motor'].get('armature', True):
                m.dof_armature[self.armature_dofs] = jr*(self.nom['gear_ratio']**2)*self.ext2
            self.v_bus = float(np.max(self.nom['v_bus']))
            self.v_nom = self.nom['v_bus'].copy()
            self.current = np.zeros(k)
            self.cut = np.zeros(k, bool)
        if self.on['friction']:
            dr = self.cfg['friction'].get('dr', {})
            m.dof_frictionloss[self.fric_dofs] = self.floss_base+self.coulomb*self._draw(dr.get('coulomb'), size=len(self.coulomb))
            m.dof_damping[self.fric_dofs] = self.damp_base+self.viscous*self._draw(dr.get('viscous'), size=len(self.viscous))
        if self.on['backlash']:
            self.backlash = self.backlash_nom*self._draw(self.cfg['backlash'].get('dr', {}).get('rad'), size=self.n)
        if self.on['thermal']:
            th, dr = self.cfg['thermal'], self.cfg['thermal'].get('dr', {})
            k = len(self.idx)
            amb = th['ambient_c']
            self.t_amb = float(self._draw(amb.get('range'), amb['nominal'])) if isinstance(amb, dict) else float(amb)
            self.thermal = Thermal2RC(self.nom['r_th_wh']*self._draw(dr.get('r_th_wh'), size=k),
                                      self.nom['r_th_ha']*self._draw(dr.get('r_th_ha'), size=k),
                                      self.nom['c_th_w']*self._draw(dr.get('c_th_w'), size=k),
                                      self.nom['c_th_h']*self._draw(dr.get('c_th_h'), size=k))
            init = th.get('initial', 'ambient')
            self.thermal.reset(self.t_amb if init == 'ambient' else float(init))
            self.t_max = self.nom['t_winding_max']
            self.t_derate = self.t_max-float(th['derate_margin_k'])
            self.alpha, self.t_ref, self.loss_factor = float(th['alpha_cu']), float(th['t_ref_c']), float(th['loss_factor'])
        if self.on['motor']:
            self._thermal_state()
        if self.on['battery']:
            bt = self.cfg['battery']
            self.v_oc = float(bt['v_oc']) if bt.get('v_oc') else self.v_bus
            ri = bt['r_internal']
            self.r_int = float(self._draw(ri.get('range'), ri['nominal'])) if isinstance(ri, dict) else float(ri)
            self.v_bus, self.i_bus = self.v_oc, 0.
        if self.on['delay']:
            pick = lambda c: int(self.rng.integers(c['range'][0], c['range'][1]+1)) if self.randomize else int(c['nominal'])
            self.k_act, self.k_obs = pick(self.cfg['delay']['action_steps']), pick(self.cfg['delay']['obs_steps'])
            self.act_line = DelayLine(self.k_act, d.ctrl[env.act].copy())
        self.sensed = env._sense()
        if self.on['delay'] and self.k_obs:
            self.obs_line = DelayLine(self.k_obs, self.sensed)
        if self.on['imu']:
            im = self.cfg['imu']
            self.gyro_bias = self.rng.normal(0, im['gyro_bias_init'], 3) if self.randomize else np.zeros(3)
            self.accel_bias = self.rng.normal(0, im['accel_bias_init'], 3) if self.randomize else np.zeros(3)
        if self.on['encoder']:
            self.enc_offset = (self.rng.uniform(*self.cfg['encoder']['offset_range'], self.n) if self.randomize
                               else np.zeros(self.n))

    # --- physics step --------------------------------------------------------------------------
    def r_hot(self):
        if not self.on['thermal']:
            return self.r0
        return self.r0*(1.+self.alpha*(self.thermal.t_w-self.t_ref))

    def _thermal_state(self):
        """Winding resistance and current limit at the present temperatures (derate, latched cut-off)."""
        self.R = self.r_hot()
        self.i_lim = self.i_peak
        if self.on['thermal']:
            self.i_lim = np.where(self.cut, 0., self.i_peak*derate(self.thermal.t_w, self.t_derate, self.t_max))

    def motor_limits(self, env, lo, hi):
        """Overwrite the force range of the motor joints with the current/voltage/thermal limits (see module doc)."""
        if not self.on['motor'] or not len(self.idx):
            return lo, hi
        d = env.data
        qd = d.qvel[self.v_adr_m]
        N = self.ratio(d.qpos[self.q_adr_m])
        kn = self.kt*N
        emf = kn*qd  # kt w_m
        R, i_lim = self.R, self.i_lim  # from the temperatures after the previous physics step (_thermal_state)
        v = self.v_bus if self.on['battery'] else self.v_nom  # with a battery every motor sees the sagging bus
        drive, brake = kn*self.eta, kn/self.eta_b
        # np.minimum/np.maximum instead of np.clip: same values, a fraction of the call overhead at 1 kHz
        hi_m = np.minimum(np.maximum((v-emf)/R, 0.), i_lim)*np.where(qd >= 0, drive, brake)
        lo_m = -np.minimum(np.maximum((v+emf)/R, 0.), i_lim)*np.where(qd <= 0, drive, brake)
        lo, hi = np.array(lo, float), np.array(hi, float)
        lo[self.idx], hi[self.idx] = lo_m, hi_m
        env.model.actuator_forcerange[self.act_m] = np.stack([lo_m, hi_m], 1)
        return lo, hi

    def command(self, env, want):
        """Servo target actually applied this physics step: delayed by k_act steps, then the backlash dead band (the
        servo has no grip on the joint while it is inside the gap: position error shrunk by half the backlash)."""
        if self.on['delay']:
            want = self.act_line.push(np.array(want, float))
        if self.on['backlash']:
            q = env.data.qpos[env.q_adr]
            e = want-q
            want = q+np.sign(e)*np.maximum(np.abs(e)-.5*self.backlash, 0.)
        return want

    def after_substep(self, env, dt):
        d = env.data
        if self.on['motor'] and len(self.idx):
            tau = d.actuator_force[self.act_m]
            qd = d.qvel[self.v_adr_m]
            N = self.ratio(d.qpos[self.q_adr_m])
            tm = np.where(tau*qd >= 0, tau/(N*self.eta), tau*self.eta_b/N)  # motor torque (drive / back-driven)
            self.current = np.abs(tm)/self.kt
            p_cu = (self.loss_factor if self.on['thermal'] else 1.)*self.current**2*self.R
            if self.on['thermal']:
                self.thermal.step(p_cu, dt, self.t_amb)
                tw = self.thermal.t_w
                self.cut = np.where(tw >= self.t_max, True, np.where(tw < self.t_derate, False, self.cut))
                self._thermal_state()
            if self.on['battery']:
                p_el = tm*N*qd+p_cu  # electrical power into each motor
                if not self.cfg['battery'].get('regen'):
                    p_el = np.maximum(p_el, 0.)
                self.i_bus = (float(np.sum(p_el))+float(self.cfg['battery']['p_aux_w']))/self.v_bus
                self.v_bus = max(self.v_oc-self.r_int*self.i_bus, .1*self.v_oc)
        if self.on['delay'] and self.k_obs:
            self.sensed = self.obs_line.push(env._sense())

    # --- observation ---------------------------------------------------------------------------
    def observe(self, env):
        s = self.sensed if (self.on['delay'] and self.k_obs) else env._sense()
        s = {k: np.array(v, float) for k, v in s.items()}
        dt = self.control_dt
        if self.on['imu']:
            im = self.cfg['imu']
            self.gyro_bias += self.rng.normal(0, im['gyro_bias_rw']*np.sqrt(dt), 3)
            self.accel_bias += self.rng.normal(0, im['accel_bias_rw']*np.sqrt(dt), 3)
            s['gyro'] = s['gyro']+self.gyro_bias+self.rng.normal(0, im['gyro_std'], 3)
            s['grav'] = s['grav']+(self.accel_bias+self.rng.normal(0, im['accel_std'], 3))/G
            if 'vel' in s:
                s['vel'] = s['vel']+self.rng.normal(0, im['vel_std'], 3)
        if self.on['encoder']:
            en = self.cfg['encoder']
            res = 2*np.pi/2**int(en['bits'])
            s['q'] = np.round((s['q']+self.enc_offset+self.rng.normal(0, en['q_std'], self.n))/res)*res
            s['qd'] = s['qd']+self.rng.normal(0, en['qd_std'], self.n)
        obs = env._compose(s)
        if self.obs_temp:
            obs = np.concatenate([obs, self.temp_obs()]).astype(np.float32)
        return obs

    def temp_obs(self):
        """Winding temperature per active joint, 0 at ambient and 1 at T_max (0 for joints without a motor model)."""
        out = np.zeros(self.n)
        out[self.idx] = (self.thermal.t_w-self.t_amb)/np.maximum(self.t_max-self.t_amb, 1.)
        return out

    def _full(self, x, fill=np.nan):
        out = np.full(self.n, fill, float)
        out[self.idx] = x
        return out

    def info(self):
        full = self._full
        out = {'action_delay_steps': self.k_act, 'obs_delay_steps': self.k_obs}
        if self.on['motor']:
            out['current'] = full(self.current)
        if self.on['thermal']:
            out.update(t_winding=full(self.thermal.t_w), t_housing=full(self.thermal.t_h), t_ambient=self.t_amb,
                       derate=full(np.where(self.cut, 0., derate(self.thermal.t_w, self.t_derate, self.t_max)), 1.),
                       cutoff=full(self.cut.astype(float), 0.) > 0)
        if self.on['battery']:
            out.update(v_bus=self.v_bus, i_bus=self.i_bus)
        return out
