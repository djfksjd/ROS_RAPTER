"""Switch walking policies mid-run on flat ground and count falls per phase.
argv: env_run_dir dr|nodr episodes v profile [seed0=5000]. env_run_dir supplies the environment (args.json, flat,
stiff ground); the policies are rev2_tr4 (terrain, evidence 181) and rev2_ct5 (high speed), each with its own frozen VecNormalize.
profile switch: tr4 at v 6 s -> ct5 at v 6 s -> ct5 at 5.5 m/s 6 s -> ct5 at v 4 s -> tr4 at v 6 s.
profile tr4_only / ct5_only: the same command sequence with one policy throughout (control)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'sim/rl')
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
d, rnd, n, v, prof = Path(sys.argv[1]), sys.argv[2] == 'dr', int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
seed0 = int(sys.argv[6]) if len(sys.argv) > 6 else 5000
cmds = [(v, 6.), (v, 6.), (5.5, 6.), (v, 4.), (v, 6.)]
pol = {'switch': ['tr4', 'ct5', 'ct5', 'ct5', 'tr4'], 'tr4_only': ['tr4']*5, 'ct5_only': ['ct5']*5}[prof]
kw = {**env_kwargs(json.loads((d/'args.json').read_text())), 'level': 0., 'episode_s': 40.}
rows = []
for ep in range(n):
    env = RunEnv(**kw, randomize=rnd, seed=seed0+ep); env.init_seed = None if rnd else seed0+ep
    P = {p: load(f'sim/rl/runs/rev2_{p}/model.zip', env) for p in ('tr4', 'ct5')}
    P['tr4'][1].reset(); env.resample_steps = 0
    raw, t, fell, vx = env._obs(), 0., None, [[] for _ in cmds]
    for i, ((c, secs), p) in enumerate(zip(cmds, pol)):
        env.command = np.array([c, 0., 0.]); model, venv = P[p]
        for k in range(int(secs/.02)):
            raw, _, term, trunc, info = env.step(model.predict(venv.normalize_obs(raw[None]), deterministic=True)[0][0])
            t += .02; vx[i].append(info['v_body'][0])
            if term:
                fell = dict(t=round(t, 2), phase=i, since_phase=round((k+1)*.02, 2)); break
        if fell: break
    rows.append(dict(seed=seed0+ep, fell=fell, vx=[round(float(np.mean(x[50:])), 2) if len(x) > 50 else None for x in vx]))
falls = [r['fell'] for r in rows if r['fell']]
print(json.dumps(dict(profile=prof, dr=rnd, v=v, n=n, falls=len(falls),
      falls_by_phase=[sum(f['phase'] == i for f in falls) for i in range(len(cmds))],
      falls_within_1s_of_switch=sum(f['since_phase'] <= 1. and pol[f['phase']] != pol[max(f['phase']-1, 0)] for f in falls),
      phases=[f'{p}@{c:g}' for (c, _), p in zip(cmds, pol)], episodes=rows)), flush=True)
