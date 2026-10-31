"""Stage gate tests of the sequential plan (docs/STAGED_PLAN.ko.md). Each gate runs a policy run directory
(model.zip, vecnorm.pkl, args.json) on fixed seeds and prints one JSON line per condition; `--gate all-before N` reruns
every earlier gate (regression rule of the plan).

stand (stage 1): zero command, the run's realism and model options, no training disturbances.
  hold     30 s, randomize off (nominal), 8 seeds: falls, largest horizontal drift (m) and |yaw| change (rad) of the base,
           hottest winding (C)
  hold_dr  30 s, randomize on (domain randomisation of the run, realism draws) but no pushes or impulses, 24 seeds
  push     base velocity kick dv (m/s) at t = 5 s toward front / back / left / right (body frame), 10 s total, randomize
           off, 8 seeds per direction and size
Fall = the env's termination (tilt, body contact, low hip). Pass rules are in the plan; this script only measures.
Usage: .venv-sim/bin/python sim/rl/stage_gates.py RUN_DIR stand [--quick]
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


GATES = {'stand': stand}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('gate', choices=list(GATES))
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    d = Path(a.run)
    for row in GATES[a.gate](d, a.quick):
        print(json.dumps({'run': d.name, 'gate': a.gate, **row}), flush=True)


if __name__ == '__main__':
    main()
