import sys, json
import numpy as np, mujoco
from pathlib import Path
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl'))
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
RUN = ROOT/'sim/rl/runs'/sys.argv[1]
def go(cmd, seed, size=2.0):
    env = RunEnv(**{**env_kwargs(json.loads((RUN/'args.json').read_text())), 'episode_s': 15., 'level': 0.}, randomize=False, seed=7); env.init_seed = seed
    model, venv = load(str(RUN/'model.zip'), env); venv.reset(); venv.reset(); d = env.data; m = env.model
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0; obs = venv.normalize_obs(env._obs()[None])
    rng = np.random.default_rng(seed); at = int((5.+rng.uniform(0., .5))/.02); sign = rng.choice([-1., 1.])
    print('mass', round(env.mass, 2), 'cmd', cmd, 'seed', seed, 'sign', sign, 'loaded at impulse:', end=' ')
    for k in range(at+40):
        if k == at:
            print(dict(env.loaded), 'phase', round(float(env.phase), 2)); env.apply_impulse('r', sign*size)
        raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None])
        if k >= at:
            R = d.xmat[env.base].reshape(3, 3); roll = np.arctan2(R[2, 1], R[2, 2]); w = R.T @ d.cvel[env.base][:3]
            cons = [(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name) for c in d.contact[:d.ncon]]
            print(f"  +{(k-at)*.02:.2f}s roll {roll:+.2f} rollrate {w[0]:+.1f} tilt {info['tilt']:.2f} z {d.qpos[2]:.2f} loaded {dict(env.loaded)} contacts {sorted(set(cons))}")
        if term: print('  TERMINATED'); break
go(4., 96009); go(7., 96004); go(4., 96001)
