"""Stage gate tests of the sequential plan (docs/STAGED_PLAN.ko.md). Each gate runs a policy run directory
(model.zip, vecnorm.pkl, args.json) on fixed seeds and prints one JSON line per condition; `--gate all-before N` reruns
every earlier gate (regression rule of the plan).

stand (stage 1): zero command, the run's realism and model options, no training disturbances.
  hold     30 s, randomize off (nominal), 8 seeds: falls, largest horizontal drift (m) and |yaw| change (rad) of the base,
           hottest winding (C)
  hold_dr  30 s, randomize on (domain randomisation of the run, realism draws) but no pushes or impulses, 24 seeds
  push     base velocity kick dv (m/s) at t = 5 s toward front / back / left / right (body frame), 10 s total, randomize
           off, 8 seeds per direction and size
getup (stage 1b): the run is a get-up policy (train_run.py --getup). Per start mode (sit, side, back) 20 fallen starts
           (randomize off): the get-up policy runs until the stand state has held 1 s without a break (success; a training
           `leak` is switched off) or 10 s pass, then the
           stand policy (--stand RUN_DIR, stage 1a) takes over for 10 s at a zero command: time to stand, falls after
           the handoff, hottest winding
walk (stage 2, defined 2026-11-03 before any walking run): randomize off unless noted, 4 seeds per condition
  track     forward command 0.3 / 0.6 / 1.0 / 1.5 m/s held 15 s from standing: mean forward body speed over 5-15 s
  turn      yaw command +-0.5 rad/s held 10 s at zero forward speed: mean yaw rate over 3-10 s, horizontal drift
  stopstart 0 (3 s) -> 1.0 (5 s) -> 0 (5 s) -> 1.0 (5 s) -> 0 (5 s): time to reach 0.8 m/s after each start, speed 2 s
            after each stop
  track_dr  1.0 m/s 15 s, randomize on, 24 seeds
Fall = the env's termination (tilt, body contact, low hip). Pass rules are in the plan; this script only measures.
Usage: .venv-sim/bin/python sim/rl/stage_gates.py RUN_DIR stand [--quick]
       .venv-sim/bin/python sim/rl/stage_gates.py RUN_DIR getup --stand sim/rl/runs/r3_s1ho [--quick]
       .venv-sim/bin/python sim/rl/stage_gates.py RUN_DIR walk [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import load  # noqa: E402
from run_env import RunEnv  # noqa: E402
from train_run import env_kwargs  # noqa: E402


def rollout(d, seed, seconds, randomize, kick=None, kick_t=5.):
    a = json.loads((d/'args.json').read_text())
    kw = {**env_kwargs(a), 'level': 0., 'episode_s': seconds+5., 'disturb_items': [], 'roll_impulse': None, 'yaw_impulse': None,
          'cmd_max': [0., 0., 0.], 'init_speed': False}
    env = RunEnv(**kw, randomize=randomize, seed=seed)
    env.init_seed = seed
    model, venv = load(str(d/'model.zip'), env)
    venv.reset()
    env.command = np.zeros(3); env.resample_steps = 10**9
    obs = venv.normalize_obs(env._obs()[None])
    dd, b = env.data, env.base
    x0 = dd.xpos[b][:2].copy()
    R = dd.xmat[b].reshape(3, 3); yaw0 = np.arctan2(R[1, 0], R[0, 0])
    drift, dyaw, fell, t_hot = 0., 0., None, 0.
    for k in range(int(seconds/.02)):
        if kick is not None and k == int(kick_t/.02):
            R = dd.xmat[b].reshape(3, 3)
            dd.qvel[:2] += (R[:2, :2] @ np.asarray(kick, float))
        raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0])
        obs = venv.normalize_obs(raw[None])
        drift = max(drift, float(np.linalg.norm(dd.xpos[b][:2]-x0)))
        R = dd.xmat[b].reshape(3, 3)
        dyaw = max(dyaw, abs(float((np.arctan2(R[1, 0], R[0, 0])-yaw0+np.pi) % (2*np.pi)-np.pi)))
        if env.rz is not None and env.rz.on.get('thermal'):
            t_hot = max(t_hot, float(np.max(env.rz.thermal.t_w)))
        if term:
            fell = round((k+1)*.02, 2)
            break
    return dict(seed=seed, t_fall=fell, drift_m=round(drift, 3), dyaw_rad=round(dyaw, 3), t_winding_max_c=round(t_hot, 1))


def summary(name, rows, **extra):
    return dict(cond=name, **extra, n=len(rows), falls=sum(r['t_fall'] is not None for r in rows),
                drift_max_m=max(r['drift_m'] for r in rows), dyaw_max_rad=max(r['dyaw_rad'] for r in rows),
                t_winding_max_c=max(r['t_winding_max_c'] for r in rows), episodes=rows)


def stand(d, quick=False):
    n1, n2, n3 = (2, 4, 2) if quick else (8, 24, 8)
    yield summary('hold', [rollout(d, 6000+i, 30., False) for i in range(n1)])
    yield summary('hold_dr', [rollout(d, 6100+i, 30., True) for i in range(n2)])
    for dv in (.3, .5, .8):
        for name, v in (('front', (dv, 0.)), ('back', (-dv, 0.)), ('left', (0., dv)), ('right', (0., -dv))):
            yield summary('push', [rollout(d, 6200+i, 10., False, kick=v) for i in range(n3)], dir=name, dv=dv)


def getup_rollout(d, stand_dir, mode, seed, t_get=10., t_stand=10.):
    a = json.loads((d/'args.json').read_text())
    g = json.loads(a['getup'])
    kw = {**env_kwargs(a), 'level': 0., 'episode_s': t_get+t_stand+5., 'disturb_items': [], 'roll_impulse': None,
          'yaw_impulse': None, 'cmd_max': [0., 0., 0.], 'init_speed': False, 'getup': {**g, 'modes': {mode: 1.}, 'leak': None}}
    env = RunEnv(**kw, randomize=False, seed=seed)
    env.init_seed = seed
    model, venv = load(str(d/'model.zip'), env)
    stand, svenv = load(str(stand_dir/'model.zip'), env)
    venv.reset()
    obs = venv.normalize_obs(env._obs()[None])
    t_up, fell, t_hot, k = None, None, 0., 0
    for k in range(int(t_get/.02)):
        raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0])
        obs = venv.normalize_obs(raw[None])
        if env.rz is not None and env.rz.on.get('thermal'):
            t_hot = max(t_hot, float(np.max(env.rz.thermal.t_w)))
        if info['getup']['success']:
            t_up = round((k+1)*.02, 2)
            break
        if term:
            break
    if t_up is not None:
        env.getup, env.hold_anchor = None, None  # handoff: stage 1a stand policy, zero command
        env.command = np.zeros(3); env.resample_steps = 10**9
        obs = svenv.normalize_obs(env._obs()[None])
        for j in range(int(t_stand/.02)):
            raw, _, term, _, _ = env.step(stand.predict(obs, deterministic=True)[0][0])
            obs = svenv.normalize_obs(raw[None])
            if env.rz is not None and env.rz.on.get('thermal'):
                t_hot = max(t_hot, float(np.max(env.rz.thermal.t_w)))
            if term:
                fell = round((j+1)*.02, 2)
                break
    return dict(seed=seed, t_up=t_up, fell_after=fell, t_winding_max_c=round(t_hot, 1))


def getup(d, quick=False, stand_dir=None):
    n = 3 if quick else 20
    for mode in ('sit', 'side', 'back'):
        rows = [getup_rollout(d, stand_dir, mode, 7000+i) for i in range(n)]
        ok = [r for r in rows if r['t_up'] is not None]
        yield dict(cond='getup', mode=mode, n=n, success=len(ok), falls_after=sum(r['fell_after'] is not None for r in ok),
                   t_up_max_s=max((r['t_up'] for r in ok), default=None),
                   t_winding_max_c=max(r['t_winding_max_c'] for r in rows), episodes=rows)


def walk_rollout(d, seed, schedule, seconds, randomize):
    """Command schedule [(t_start, (vx, vy, yaw))]; returns per-step time, body-frame forward speed, yaw rate, xy."""
    a = json.loads((d/'args.json').read_text())
    kw = {**env_kwargs(a), 'level': 0., 'episode_s': seconds+5., 'disturb_items': [], 'roll_impulse': None, 'yaw_impulse': None,
          'cmd_max': [0., 0., 0.], 'init_speed': False}
    env = RunEnv(**kw, randomize=randomize, seed=seed)
    env.init_seed = seed
    model, venv = load(str(d/'model.zip'), env)
    venv.reset()
    env.resample_steps = 10**9
    dd, b = env.data, env.base
    log, fell = [], None
    for k in range(int(seconds/.02)):
        t = k*.02
        env.command = np.array(max((c for c in schedule if c[0] <= t), key=lambda c: c[0])[1], float)
        if k == 0:
            obs = venv.normalize_obs(env._obs()[None])
        raw, _, term, _, _ = env.step(model.predict(obs, deterministic=True)[0][0])
        obs = venv.normalize_obs(raw[None])
        R = dd.xmat[b].reshape(3, 3)
        v_body = R.T @ dd.qvel[:3]
        log.append((t+.02, float(v_body[0]), float((R.T @ dd.qvel[3:6])[2]), *map(float, dd.xpos[b][:2])))
        if term:
            fell = round(t+.02, 2)
            break
    return fell, np.array(log)


def walk(d, quick=False):
    n, ndr = (2, 4) if quick else (4, 24)
    mean = lambda L, c, t0, t1: float(np.mean(L[(L[:, 0] >= t0) & (L[:, 0] < t1), c])) if len(L) and L[-1, 0] >= t1 else None
    for v in (.3, .6, 1., 1.5):
        rows = []
        for i in range(n):
            fell, L = walk_rollout(d, 6300+i, [(0., (v, 0., 0.))], 15., False)
            rows.append(dict(seed=6300+i, t_fall=fell, v_mean=mean(L, 1, 5., 15.)))
        yield dict(cond='track', cmd=v, n=n, falls=sum(r['t_fall'] is not None for r in rows), episodes=rows)
    for w in (.5, -.5):
        rows = []
        for i in range(n):
            fell, L = walk_rollout(d, 6400+i, [(0., (0., 0., w))], 10., False)
            rows.append(dict(seed=6400+i, t_fall=fell, yaw_rate_mean=mean(L, 2, 3., 10.),
                             drift_m=round(float(np.max(np.linalg.norm(L[:, 3:5]-L[0, 3:5], axis=1))), 3) if len(L) else None))
        yield dict(cond='turn', cmd=w, n=n, falls=sum(r['t_fall'] is not None for r in rows), episodes=rows)
    sched = [(0., (0., 0., 0.)), (3., (1., 0., 0.)), (8., (0., 0., 0.)), (13., (1., 0., 0.)), (18., (0., 0., 0.))]
    rows = []
    for i in range(n):
        fell, L = walk_rollout(d, 6500+i, sched, 23., False)
        reach = []
        for t0 in (3., 13.):
            hit = L[(L[:, 0] >= t0) & (L[:, 0] < t0+5.) & (L[:, 1] >= .8)] if len(L) else []
            reach.append(round(float(hit[0, 0]-t0), 2) if len(hit) else None)
        stop = [mean(L, 1, t0+2., t0+3.) for t0 in (8., 18.)]
        rows.append(dict(seed=6500+i, t_fall=fell, t_reach_0p8=reach, speed_2s_after_stop=stop))
    yield dict(cond='stopstart', n=n, falls=sum(r['t_fall'] is not None for r in rows), episodes=rows)
    rows = []
    for i in range(ndr):
        fell, L = walk_rollout(d, 6600+i, [(0., (1., 0., 0.))], 15., True)
        rows.append(dict(seed=6600+i, t_fall=fell, v_mean=mean(L, 1, 5., 15.)))
    yield dict(cond='track_dr', cmd=1., n=ndr, falls=sum(r['t_fall'] is not None for r in rows), episodes=rows)


GATES = {'stand': stand, 'getup': getup, 'walk': walk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('gate', choices=list(GATES))
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--stand', help='stage 1a stand policy run directory (getup gate handoff)')
    a = ap.parse_args()
    d = Path(a.run)
    for row in (GATES[a.gate](d, a.quick, Path(a.stand)) if a.gate == 'getup' else GATES[a.gate](d, a.quick)):
        print(json.dumps({'run': d.name, 'gate': a.gate, **row}), flush=True)


if __name__ == '__main__':
    main()
