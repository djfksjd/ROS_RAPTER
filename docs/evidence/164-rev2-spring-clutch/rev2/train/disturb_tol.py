"""Disturbance tolerance of a frozen policy (no training): one impulse of a given kind and size at a seeded time between
5.0 and 5.5 s of a deterministic rollout (no other randomisation); a trial fails if the robot falls within the next 5 s.
Kinds: yaw / pitch / roll torque impulse on the trunk (N·m·s, 40 ms), lateral and fore-aft push on the trunk (N·s, 100 ms).
Usage (project root): .venv-sim/bin/python disturb_tol.py <run> <out.jsonl> [n_trials=16] [overrides-json]"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl'))
RUN = ROOT/'sim/rl/runs'/sys.argv[1]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 16; OVER = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
GRID = dict(yaw=(.5, 1., 1.5, 2., 2.5), pitch=(1., 2., 4.), roll=(1., 2., 4.), lateral=(2.5, 5., 10.), forward=(5., 10.), backward=(5., 10.))


def trial(task):
    kind, size, cmd, seed = task
    from evaluate import load
    from run_env import RunEnv
    from train_run import env_kwargs
    env = RunEnv(**{**env_kwargs(json.loads((RUN/'args.json').read_text())), 'episode_s': 15., 'level': 0., **OVER}, randomize=False, seed=7); env.init_seed = seed
    model, venv = load(str(RUN/'model.zip'), env); venv.reset(); venv.reset(); d = env.data
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0; obs = venv.normalize_obs(env._obs()[None])
    rng = np.random.default_rng(seed); at = int((5.+rng.uniform(0., .5))/.02); sign = rng.choice([-1., 1.]); fell = False; t_fall = None; v = []
    for k in range(int(10.5/.02)):
        if k == at:
            if kind in ('yaw', 'pitch', 'roll'):
                env.apply_impulse(dict(yaw='z', pitch='y', roll='r')[kind], sign*size)
            else:
                R = d.xmat[env.base].reshape(3, 3); ax = R[:, 1]*sign if kind == 'lateral' else R[:, 0]*(1. if kind == 'forward' else -1.)
                env.apply_force(env.base, ax*size/.1, .1)
        raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None]); v.append(float(info['v_body'][0]))
        if term:
            fell = True; t_fall = k*.02-at*.02; break
    return dict(kind=kind, size=size, cmd=cmd, seed=seed, fell=fell, before=bool(fell and t_fall < 0), t_after=t_fall, speed_end=float(np.mean(v[-50:])))


if __name__ == '__main__':
    tasks = [(k, s, c, 96001+i) for k, g in GRID.items() for s in g for c in (2., 4., 7.) for i in range(N)]; t0 = time.time()
    with Pool(8) as pool, open(sys.argv[2], 'w') as f:
        for r in pool.imap_unordered(trial, tasks, chunksize=2):
            f.write(json.dumps(r)+'\n')
    R = [json.loads(l) for l in open(sys.argv[2])]
    for k, g in GRID.items():
        print(k, {s: [f"{sum(r['fell'] for r in R if r['kind'] == k and r['size'] == s and r['cmd'] == c)}/{N}" for c in (2., 4., 7.)] for s in g})
    print('done', len(R), round(time.time()-t0), 's')
