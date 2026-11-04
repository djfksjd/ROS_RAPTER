"""Frozen-model, fully realistic keyframe jump probes. No learned policy or reward changes."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
from run_env import RunEnv
from train_run import env_kwargs

E = Path(__file__).resolve().parent
RUN = ROOT/'sim/rl/runs/l80_r1_6M'
NAMES = ['crouch_hip', 'crouch_knee', 'crouch_tail', 'push_hip', 'push_knee', 'push_tail',
         'land_hip', 'land_knee', 'land_tail', 'crouch_s', 'push_s', 'extend_s', 'land_s']
BOUNDS = np.array([[-1.1, -.2], [1.05, 2.0], [-.7, .7], [-.9, .3], [.62, 1.05], [-.7, .7],
                   [-1., .1], [.8, 1.8], [-.7, .7], [.12, .5], [.04, .24], [.04, .3], [.04, .24]])
GUESS = np.array([-.75, 1.6, .0, -.45, .65, .0, -.55, 1.05, .0, .3, .10, .10, .12])
HORIZON = 3.6


def make(seed=None, terrain=None):
    args = json.loads((RUN/'args.json').read_text())
    kw = {**env_kwargs(args), 'level': 0., 'episode_s': 15., 'disturb_items': [], 'roll_impulse': None,
          'yaw_impulse': None, 'cmd_max': [0., 0., 0.], 'cmd_mix': None, 'init_speed': False, 'terrain': terrain}
    env = RunEnv(**kw, randomize=False, seed=0)
    env.init_seed = seed
    return env


def encode(values):
    return np.clip(2*(values-BOUNDS[:, 0])/(BOUNDS[:, 1]-BOUNDS[:, 0])-1, -1., 1.)


def decode(x):
    return BOUNDS[:, 0]+(np.clip(x, -1., 1.)+1)/2*(BOUNDS[:, 1]-BOUNDS[:, 0])


def pose(env, values):
    q = env.q0.copy()
    for i, name in enumerate(env.active):
        if 'hip_pitch' in name:
            q[i] = values[0]
        elif 'knee_pitch' in name:
            q[i] = values[1]
        elif name == 'tail_pitch_joint':
            q[i] = values[2]
    pi = env.policy_idx
    return np.clip((q[pi]-env.q0[pi])/env.scale[pi], -1., 1.)


def keyframes(env, x):
    p = decode(x)
    times = np.r_[0., .2, .2+np.cumsum(p[9:])]
    crouch, push, land = (pose(env, p[k:k+3]) for k in (0, 3, 6))
    values = np.array([np.zeros(len(env.policy_idx)), np.zeros(len(env.policy_idx)), crouch, push, push, land])
    return times, values


def action_at(t, times, values):
    return np.array([np.interp(t, times, values[:, i]) for i in range(values.shape[1])])


class Recorder:
    """Observe every physics step after the original realism update; never advances sensors or RNG."""
    def __init__(self, env):
        self.env = env
        self.rows = {k: [] for k in ('t', 'xyz', 'velocity', 'tilt', 'feet_n', 'feet_force_n', 'vertical_force_n',
                     'body_hit', 'q', 'joint_tau_nm', 'drive_tau_nm', 'limit_nm', 'current_a', 'current_limit_a',
                     'current_peak_a', 'overcurrent')}
        self.original_after, self.original_clamp = env.rz.after_substep, env._clamp
        self.lo, self.hi = None, None

    def clamp(self):
        self.lo, self.hi = self.original_clamp()
        return self.lo, self.hi

    def sample(self, env, dt):
        self.original_after(env, dt)
        d, m, rz = env.data, env.model, env.rz
        count, force = np.zeros(2, int), np.zeros(2)
        vertical, hit = 0., False
        for i in range(d.ncon):
            contact = d.contact[i]
            other = contact.geom2 if contact.geom1 in env.ground else contact.geom1 if contact.geom2 in env.ground else None
            if other is None:
                continue
            side = next((j for j, name in enumerate(('left', 'right')) if other in env.foot_geoms[name]), None)
            if side is None:
                hit = True
                continue
            f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f)
            count[side] += 1
            force[side] += abs(f[0])
            vertical += abs(float((np.asarray(contact.frame).reshape(3, 3).T @ f[:3])[2]))
        idx = rz.idx
        joint_tau = d.actuator_force[rz.act_m].copy()
        drive_tau = joint_tau.copy()
        if rz.on['backlash'] and (rz.bl_m >= 0).any():
            drive_tau = np.where(rz.bl_m >= 0, rz.bl_tau[rz.bl_m], drive_tau)
        limit = np.where(drive_tau >= 0, self.hi[idx], -self.lo[idx])
        values = dict(t=float(d.time), xyz=d.xpos[env.base].copy(), velocity=d.qvel[:3].copy(),
            tilt=float(np.arccos(np.clip(d.xmat[env.base].reshape(3, 3)[2, 2], -1, 1))),
            feet_n=count, feet_force_n=force, vertical_force_n=vertical, body_hit=hit,
            q=d.qpos[env.q_adr[idx]].copy(), joint_tau_nm=joint_tau, drive_tau_nm=drive_tau,
            limit_nm=limit.copy(), current_a=rz.current.copy(), current_limit_a=rz.i_lim.copy(),
            current_peak_a=rz.i_peak.copy(), overcurrent=rz.overcurrent.copy())
        for k, value in values.items():
            self.rows[k].append(value)

    def __enter__(self):
        self.env._clamp, self.env.rz.after_substep = self.clamp, self.sample
        return self

    def __exit__(self, *_):
        self.env._clamp, self.env.rz.after_substep = self.original_clamp, self.original_after

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.rows.items()}


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def longest(mask):
    return max((b-a for a, b in intervals(mask)), default=0)*.001


def measurements(env, arrays, fallen, launch_after=.2):
    a = arrays
    t, z = a['t'], a['xyz'][:, 2]
    origin_i = min(np.searchsorted(t, .2), len(t)-1)
    origin = a['xyz'][origin_i]
    support = (a['feet_force_n'] > .05*env.weight).all(axis=1)
    off = (a['feet_n'] == 0).all(axis=1)
    events = []
    for start, stop in intervals(off):
        if stop-start < 20 or t[start] < launch_after or a['velocity'][start, 2] <= .05:
            continue
        if not any(b-a >= 20 for a, b in intervals(support[:start])):
            continue
        landed = bool(stop < len(t))
        end = stop if landed else stop-1
        events.append(dict(start=int(start), stop=int(stop), takeoff_s=float(t[start]),
            touchdown_s=float(t[stop]) if landed else None, flight_s=float((stop-start)*.001),
            apex_rise_m=float(z[start:stop].max()-origin[2]), apex_above_takeoff_m=float(z[start:stop].max()-z[start]),
            distance_m=float(a['xyz'][end, 0]-a['xyz'][start, 0]), landing_offset_m=float(a['xyz'][end, 0]-origin[0]),
            landed=landed))
    event = events[0] if events else None
    stand = support & (a['tilt'] < .3) & (z > .8*env.h_nom) & (np.linalg.norm(a['velocity'], axis=1) < .5)
    post = slice(event['stop'], None) if event and event['landed'] else slice(0, 0)
    stable_s = longest(stand[post])
    no_fall_s = float(t[-1]-event['touchdown_s']) if event and event['landed'] and fallen is None else 0.
    max_force = float(a['feet_force_n'][post].sum(axis=1).max()) if len(a['t'][post]) else None
    out = dict(event=event, events=events, t_fall=fallen, alive_s=float(t[-1]), initial_base_z_m=float(origin[2]),
               stand_after_landing_s=stable_s, held_after_landing_without_fall_s=no_fall_s,
               landing_force_peak_n=max_force, landing_force_peak_bw=max_force/env.weight if max_force is not None else None,
               max_upward_velocity=float(a['velocity'][:, 2].max()), max_body_tilt=float(a['tilt'].max()),
               sample_count=len(t), body_contact_samples=int(a['body_hit'].sum()))
    return out


def score(result, mode):
    e = result['event']
    if e is None:
        return .2*max(0., result['max_upward_velocity'])
    metric = 10*max(e['apex_rise_m'], 0.)+2*e['flight_s']
    if mode == 'forward':
        metric += 5*max(e['distance_m'], 0.)
    else:
        metric *= np.exp(-(e['landing_offset_m']/.25)**2)
    settled = result['t_fall'] is None and result['stand_after_landing_s'] >= 1.
    valid_target = e['distance_m'] > 0 if mode == 'forward' else abs(e['landing_offset_m']) <= .25
    return float(metric+5*min(result['held_after_landing_without_fall_s'], 1.)
                 +10*min(result['stand_after_landing_s'], 1.)+50*int(settled and valid_target))


def reset_stationary(env, seed=None):
    """init_seed also adds vx in RunEnv; keep only pose jitter for a standing-start probe."""
    env.init_seed = seed
    env.reset(seed=0 if seed is None else seed)
    if seed is not None:
        env.data.qvel[:6] = 0.
        mujoco.mj_forward(env.model, env.data)
        env.rz.reset(env)  # delay/sensor buffers must reflect the actual zero-velocity start


def rollout(env, x, seed=None, capture=False):
    reset_stationary(env, seed)
    env.command = np.zeros(3); env.resample_steps = 0
    times, values = keyframes(env, x)
    fallen = None
    with Recorder(env) as recorder:
        for k in range(round(HORIZON/.02)):
            _, _, terminated, _, _ = env.step(action_at(k*.02, times, values))
            if terminated or recorder.rows['body_hit'][-20:].count(True):
                fallen = float(env.data.time)
                break
    arrays = recorder.arrays()
    result = measurements(env, arrays, fallen, launch_after=times[2])
    result['parameters'] = dict(zip(NAMES, decode(x).tolist()))
    return result, arrays if capture else None


def loads(env, arrays, event=None):
    mask = np.ones(len(arrays['t']), bool)
    if event:
        mask = (arrays['t'] >= .2) & (arrays['t'] < event['takeoff_s'])
    out = {}
    for j, i in enumerate(env.rz.idx):
        tau = np.abs(arrays['drive_tau_nm'][:, j]); lim = arrays['limit_nm'][:, j]
        current = arrays['current_a'][:, j]
        usable = mask & (lim > 1e-6)
        peak = int(np.argmax(tau))
        out[env.active[i]] = dict(peak_drive_nm=float(tau[peak]), limit_at_peak_nm=float(lim[peak]),
            peak_joint_nm=float(np.abs(arrays['joint_tau_nm'][:, j]).max()),
            peak_current_a=float(current.max()), current_peak_limit_a=float(arrays['current_peak_a'][0, j]),
            peak_current_over_limit=float(np.max(current/np.maximum(arrays['current_limit_a'][:, j], 1e-9))),
            takeoff_saturation_share=float(np.mean(tau[usable] >= .98*lim[usable])) if usable.any() else 0.,
            current_over_peak=float(np.max(current/arrays['current_peak_a'][:, j])),
            overcurrent_samples=int(arrays['overcurrent'][:, j].sum()))
    return out


def save_trace(path, arrays, env):
    np.savez_compressed(path, **arrays, joint_names=np.array([env.active[i] for i in env.rz.idx]), weight_n=env.weight)
