"""Winding temperature / derate probe (evaluate_run episode pattern), realism from args.json; argv: run_dir dr|nodr episodes."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'sim/rl')
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
d = Path(sys.argv[1]); rnd = sys.argv[2] == 'dr'; n = int(sys.argv[3]); secs = 60. if rnd else 20.
kw = {**env_kwargs(json.loads((d/'args.json').read_text())), 'level': 0., 'episode_s': secs+5.}
for vx in (2., 4., 5.5, 7.):
    rows = []
    for ep in range(n):
        env = RunEnv(**kw, randomize=rnd, seed=5000+ep); env.init_seed = None if rnd else 5000+ep
        model, venv = load(str(d/'model.zip'), env)
        venv.reset(); env.command = np.array([vx, 0., 0.]); env.resample_steps = 0
        obs = venv.normalize_obs(env._obs()[None])
        tw, der, cut, vxs, fell = 0., 1., 0, [], False
        for t in range(int(secs/.02)):
            raw, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0][0])
            obs = venv.normalize_obs(raw[None])
            if term: fell = True; break
            r = info.get('realism', {})
            w = np.asarray(r.get('t_winding', []), float); tw = max(tw, float(np.nanmax(w))) if w.size else tw
            dr = np.asarray(r.get('derate', []), float); der = min(der, float(np.nanmin(dr))) if dr.size else der
            c = np.asarray(r.get('cutoff', []), float); cut += int(np.nansum(c)) if c.size else 0
            vxs.append(info['v_body'][0])
        rows.append(dict(fell=fell, vx=float(np.mean(vxs[100:])) if len(vxs) > 100 else np.nan, tw=tw, der=der, cut=cut))
    print(json.dumps(dict(run=d.name, cmd=vx, dr=rnd, falls=sum(r['fell'] for r in rows), n=n,
          vx=round(float(np.nanmean([r['vx'] for r in rows])), 2), t_winding_max_c=round(max(r['tw'] for r in rows), 1),
          min_derate=round(min(r['der'] for r in rows), 3), cutoff_steps=sum(r['cut'] for r in rows))), flush=True)
