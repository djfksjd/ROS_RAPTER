"""Design-freeze plan F2 (docs/DESIGN_FREEZE_PLAN.ko.md): can R-03 run at all? A hand-structured running controller
(no policy) whose parameters are searched with the cross-entropy method, then replayed through RunEnv.step.

Controller (targets for the PD servos every CONTROL_DT, as a policy would set them):
  clock gait at stride frequency f, duty d per leg, legs half a stride apart (open-loop timing);
  stance: hip pitch sweeps touchdown -> lift-off angle, plus body pitch PD; knee at k_st, extended by `push` over the
          stance; swing: hip pitch returns with a cosine profile to the touchdown angle moved by k_v (v - v_cmd)
          (Raibert foot placement); knee folds by `fold` (sine); hip roll: stance width r0 plus body roll and lateral
          velocity feedback; hip yaw 0; tail pitch/yaw PD on body pitch rate / yaw rate.
Search: start standing at the nominal pose with the base at min(v_cmd, 2) m/s, the command ramps at 2 m/s^2 to v_cmd,
then 3 s at v_cmd; score = time alive + speed tracking against the ramped command.
The search steps the physics directly with the env's motor limits (RunEnv._clamp: current, T-N curve, voltage; the
realism delays, sensor noise and heating are not stepped there). The best mean is then replayed through RunEnv.step
(all realism layers on, randomize off) for the reported numbers.
A failed search is "not found within the budget", not "impossible" (plan section 2).
Usage: .venv-sim/bin/python docs/evidence/194-gait-feasibility/gait_cem.py --v 4 6 8 11.1 [--iters 30 --pop 64]
       [--override '{"rev3_config": "sim/rl/rev3_fold.json"}'] [--tag current]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'sim/rl'))
from run_env import RunEnv, PHYSICS_DT  # noqa: E402
from raptor_env import CONTROL_DT  # noqa: E402
from train_run import env_kwargs  # noqa: E402

# name: (lo, hi) of each controller parameter; the search works in [-1, 1] mapped linearly
CLOCK = dict(f=(1.5, 4.5), duty=(.2, .5), hp_td=(-1.5, -.2), hp_lo=(-.7, .7), k_st=(.7, 1.9), push=(0., .6),
              fold=(0., 1.2), k_v=(-.3, .3), kp_pitch=(-2., 2.), kd_pitch=(-.4, .4), k_roll=(-1., 1.), kd_roll=(-.3, .3),
              r0=(-.2, .2), tail_kp=(-2., 2.))
# event: stance/swing switched by foot contact and a stance time (Raibert): touchdown hip angle from the neutral point
# v Ts / 2 plus k_v (v - v_cmd) over the leg length L, stance sweep at v / L plus body pitch PD
EVENT = dict(Ts=(.06, .4), Tsw=(.1, .5), x0=(-1.2, .2), k_v=(-.2, .2), L=(.4, 1.2), k_st=(.7, 1.9), push=(0., .6),
             fold=(0., 1.2), kp_pitch=(-2., 2.), kd_pitch=(-.4, .4), k_roll=(-1., 1.), kd_roll=(-.3, .3), r0=(-.2, .2),
             tail_kp=(-2., 2.))
PARAMS, NAMES, MODE = CLOCK, list(CLOCK), 'clock'


def set_mode(mode):
    global PARAMS, NAMES, MODE
    MODE, PARAMS = mode, {'clock': CLOCK, 'event': EVENT}[mode]
    NAMES = list(PARAMS)
V0, ACC, HOLD = 2., 2., 3.
_env, OVERRIDE = None, {}


def make(override=None):
    a = json.loads((ROOT/'sim/rl/runs/r3_s1ho/args.json').read_text())
    kw = {**env_kwargs(a), 'disturb_items': [], 'roll_impulse': None, 'yaw_impulse': None, 'cmd_max': [0., 0., 0.],
          'init_speed': False, 'action_lpf': None,
          'scale_override': {'hip_roll': .5, 'hip_yaw': .5, 'hip_pitch': 1.45, 'knee_pitch': 1.3}, **(override or {})}
    return RunEnv(**kw, randomize=False, seed=1)


def unpack(x):
    x = np.clip(x, -1, 1)
    return {n: lo+(v+1)/2*(hi-lo) for n, v, (lo, hi) in zip(NAMES, x, PARAMS.values())}


class Controller:
    def __init__(self, e, p, v_cmd):
        self.e, self.p, self.v = e, p, v_cmd
        idx = {n: i for i, n in enumerate(e.active)}
        self.j = {s: {k: idx[f'{s}_{k}_joint'] for k in ('hip_roll', 'hip_yaw', 'hip_pitch', 'knee_pitch')} for s in ('left', 'right')}
        self.tail = {k: idx[f'tail_{k}_joint'] for k in ('yaw', 'pitch')}
        self.t = 0.

    def targets(self):
        e, p, d = self.e, self.p, self.e.data
        R = d.xmat[e.base].reshape(3, 3)
        pitch = float(np.arctan2(-R[2, 0], R[2, 2]))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        w = R.T@d.qvel[3:6]                       # body angular velocity
        vx, vy = float(d.qvel[0]), float(d.qvel[1])
        tgt = e.q0.copy()
        for k, s in enumerate(('left', 'right')):
            sg = 1. if s == 'left' else -1.
            ph = (self.t*p['f'] + .5*k) % 1.
            if ph < p['duty']:                    # stance
                u = ph/p['duty']
                hp = p['hp_td']+(p['hp_lo']-p['hp_td'])*u + p['kp_pitch']*pitch + p['kd_pitch']*w[1]
                kn = p['k_st']-p['push']*u
            else:                                 # swing
                u = (ph-p['duty'])/(1-p['duty'])
                td = p['hp_td']-p['k_v']*(vx-self.v)
                hp = p['hp_lo']+(td-p['hp_lo'])*(1-np.cos(np.pi*u))/2
                kn = p['k_st']+p['fold']*np.sin(np.pi*u)
            tgt[self.j[s]['hip_pitch']] = hp
            tgt[self.j[s]['knee_pitch']] = kn
            tgt[self.j[s]['hip_roll']] = sg*p['r0'] + p['k_roll']*roll + p['kd_roll']*vy
            tgt[self.j[s]['hip_yaw']] = 0.
        tgt[self.tail['pitch']] = p['tail_kp']*pitch
        tgt[self.tail['yaw']] = -.2*w[2]
        return np.clip(tgt, e.lo, e.hi)


class EventController(Controller):
    def __init__(self, e, p, v_cmd):
        super().__init__(e, p, v_cmd)
        self.state = {'left': ['stance', 0., p['x0']], 'right': ['swing', 0., p['x0']]}  # phase, time in it, touchdown hip

    def targets(self):
        e, p, d = self.e, self.p, self.e.data
        R = d.xmat[e.base].reshape(3, 3)
        pitch = float(np.arctan2(-R[2, 0], R[2, 2]))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        w = R.T@d.qvel[3:6]
        vx, vy = float(d.qvel[0]), float(d.qvel[1])
        feet, _ = e._contacts()
        tgt = e.q0.copy()
        for s in ('left', 'right'):
            sg = 1. if s == 'left' else -1.
            st = self.state[s]
            st[1] += CONTROL_DT
            if st[0] == 'stance' and st[1] >= p['Ts']:
                st[0], st[1] = 'swing', 0.
            elif st[0] == 'swing' and st[1] > .5*p['Tsw'] and feet[s] > .15*e.weight:
                st[0], st[1] = 'stance', 0.
            if st[0] == 'swing':
                td = p['x0']-(vx*p['Ts']/2+p['k_v']*(vx-self.v))/p['L']
                st[2] = td
                u = min(st[1]/p['Tsw'], 1.)
                hp = td
                kn = p['k_st']+p['fold']*np.sin(np.pi*u)
            else:
                u = min(st[1]/p['Ts'], 1.)
                hp = st[2]+vx*st[1]/p['L'] + p['kp_pitch']*pitch + p['kd_pitch']*w[1]
                kn = p['k_st']-p['push']*u
            tgt[self.j[s]['hip_pitch']] = hp
            tgt[self.j[s]['knee_pitch']] = kn
            tgt[self.j[s]['hip_roll']] = sg*p['r0'] + p['k_roll']*roll + p['kd_roll']*vy
            tgt[self.j[s]['hip_yaw']] = 0.
        tgt[self.tail['pitch']] = p['tail_kp']*pitch
        tgt[self.tail['yaw']] = -.2*w[2]
        return np.clip(tgt, e.lo, e.hi)


def controller(e, x, v_cmd):
    return (EventController if MODE == 'event' else Controller)(e, unpack(x), v_cmd)


def start(e, v_cmd):
    e.reset(seed=1)
    e.data.qvel[0] = min(v_cmd, V0)


def horizon(v_cmd):
    return max(v_cmd-V0, 0.)/ACC+HOLD


def command(v_cmd, t):
    return min(v_cmd, min(v_cmd, V0)+ACC*t)


def fallen(e):
    d = e.data
    R = d.xmat[e.base].reshape(3, 3)
    return d.xpos[e.base][2] < .45*e.h_nom/.825 or R[2, 2] < np.cos(.8) or not np.isfinite(d.qpos).all()


def rollout(e, x, v_cmd):
    start(e, v_cmd)
    T = horizon(v_cmd)
    c, m, d = controller(e, x, v_cmd), e.model, e.data
    n_ctrl = int(round(CONTROL_DT/PHYSICS_DT))
    prev = d.qpos[e.q_adr].copy()
    alive, track = 0., 0.
    for k in range(int(T/CONTROL_DT)):
        c.v = command(v_cmd, c.t)
        tgt = c.targets()
        for i in range(n_ctrl):  # target ramp over the control step, as RunEnv target_ramp
            d.ctrl[e.act] = prev+(tgt-prev)*(i+1)/n_ctrl
            e._clamp(); mujoco.mj_step(m, d)
        prev = tgt
        c.t += CONTROL_DT
        if fallen(e):
            break
        alive += CONTROL_DT
        track += CONTROL_DT*max(-1., 1-abs(float(d.qvel[0])-c.v)/c.v)
    return alive+track, alive


def replay(e, x, v_cmd, T=None):
    """Through RunEnv.step (realism delays, sensor noise, heating, the env's own target ramp). Actions are the
    controller targets expressed in the env's action units; the controller reads the true state (no estimator)."""
    start(e, v_cmd)
    T = T or horizon(v_cmd)+7.
    c, d = controller(e, x, v_cmd), e.data
    pi = e.policy_idx
    vs, sat, tau = [], [], []
    alive = 0.
    for k in range(int(T/CONTROL_DT)):
        c.v = command(v_cmd, c.t)
        a = np.clip((c.targets()[pi]-e.q0[pi])/e.scale[pi], -1, 1)
        e.step(a)
        c.t += CONTROL_DT
        if fallen(e):
            break
        alive += CONTROL_DT
        vs.append(float(d.qvel[0]))
        f = d.actuator_force[e.act]
        lim = np.maximum(np.abs(e.model.actuator_forcerange[e.act, 0]), np.abs(e.model.actuator_forcerange[e.act, 1]))
        sat.append((lim > 1e-6) & (np.abs(f) >= .98*lim))
        tau.append(np.abs(f))
    names = [n.replace('_joint', '') for n in e.active]
    temp = None
    if e.rz is not None and hasattr(e.rz, 'temps'):
        temp = round(float(np.max(e.rz.temps())), 1)
    out = dict(v_cmd=v_cmd, alive_s=round(alive, 2), mean_speed=round(float(np.mean(vs[len(vs)//2:])), 2) if vs else None,
               winding_max_c=temp)
    if sat:
        S, Tq = np.array(sat), np.array(tau)
        out['saturated_frac'] = {n: round(float(S[:, i].mean()), 2) for i, n in enumerate(names) if ('left' in n or 'tail' in n) and 'ankle' not in n}
        out['rms_nm'] = {n: round(float(np.sqrt((Tq[:, i]**2).mean())), 1) for i, n in enumerate(names) if ('left' in n or 'tail' in n) and 'ankle' not in n}
    return out


def _init(override, mode):
    global _env
    set_mode(mode)
    _env = make(override)


def _eval(args):
    x, v = args
    return rollout(_env, x, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v', type=float, nargs='+', default=[4., 6., 8., 11.1])
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--pop', type=int, default=64)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--override', help='JSON RunEnv overrides (shape variant)')
    ap.add_argument('--tag', default='current')
    ap.add_argument('--ctrl', choices=['clock', 'event'], default='clock')
    a = ap.parse_args()
    override = json.loads(a.override) if a.override else {}
    set_mode(a.ctrl)
    rng = np.random.default_rng(0)
    # start near a plausible running gait: stance hip -1.0 -> 0.1, knee 1.1, fold 0.6, 2.5 Hz, duty 0.35
    guess = dict(f=2.5, duty=.35, hp_td=-1.0, hp_lo=.1, k_st=1.1, push=.2, fold=.6, k_v=.1, kp_pitch=0., kd_pitch=0.,
                 k_roll=0., kd_roll=0., r0=0., tail_kp=0.) if a.ctrl == 'clock' else \
        dict(Ts=.15, Tsw=.25, x0=-.5, k_v=.05, L=.75, k_st=1.1, push=.2, fold=.6, kp_pitch=0., kd_pitch=0., k_roll=0.,
             kd_roll=0., r0=0., tail_kp=0.)
    mu = np.array([2*(guess[n]-lo)/(hi-lo)-1 for n, (lo, hi) in PARAMS.items()])
    sd = np.full(len(NAMES), .5)
    res = dict(tag=a.tag, ctrl=a.ctrl, override=override, params=PARAMS, speeds={})
    with Pool(a.procs, initializer=_init, initargs=(override, a.ctrl)) as pool:
        for v in a.v:
            hist = []
            sd = np.maximum(sd, .3)                # re-open the search at each speed, warm-started from the last mean
            for it in range(a.iters):
                X = np.clip(mu+sd*rng.standard_normal((a.pop, len(NAMES))), -1, 1)
                r = pool.map(_eval, [(x, v) for x in X])
                sc = np.array([s for s, _ in r]); al = np.array([t for _, t in r])
                el = X[np.argsort(-sc)[:a.pop//8]]
                mu, sd = el.mean(0), np.maximum(el.std(0), .03)
                hist.append(dict(it=it, best=round(float(sc.max()), 2), mean=round(float(sc.mean()), 2), alive_max=round(float(al.max()), 2)))
                print(v, json.dumps(hist[-1]), flush=True)
            rep = replay(make(override), mu, v)
            print(v, 'replay', json.dumps(rep), flush=True)
            res['speeds'][str(v)] = dict(hist=hist, mu=mu.round(4).tolist(), params={k: round(float(u), 3) for k, u in unpack(mu).items()},
                                         replay=rep)
            (OUT/'eval').mkdir(exist_ok=True)
            (OUT/f'eval/gait_{a.tag}_{a.ctrl}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
