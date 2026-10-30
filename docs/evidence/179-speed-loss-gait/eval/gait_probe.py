"""Gait metrics on flat stiff ground, nominal (no randomization or impulses). argv: run_dir episodes [cmds...].
Per command: speed, flight fraction (both feet unloaded), step frequency (touchdowns per second, both feet), stride
length per touchdown (speed / step frequency), swing foot clearance (peak foot-link height above its stance height),
hip height, mechanical power and the policy action magnitude. First 4 s of each 20 s run are skipped."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'sim/rl')
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
d, n = Path(sys.argv[1]), int(sys.argv[2])
cmds = [float(c) for c in sys.argv[3:]] or [2., 4., 5.5, 7.]
kw = {**env_kwargs(json.loads((d/'args.json').read_text())), 'level': 0., 'episode_s': 25.}
for vx in cmds:
    M = {k: [] for k in ('vx', 'flight', 'step_hz', 'stride', 'clear', 'hip', 'power', 'act', 'fell')}
    for ep in range(n):
        env = RunEnv(**kw, randomize=False, seed=5000+ep); env.init_seed = 5000+ep
        model, venv = load(str(d/'model.zip'), env)
        venv.reset(); env.command = np.array([vx, 0., 0.]); env.resample_steps = 0
        raw = env._obs(); s = {k: [] for k in ('vx', 'flight', 'hip', 'power', 'act')}
        prev, td, fz, stance_z, clear, fell = None, 0, {'left': [], 'right': []}, {}, [], False
        for t in range(1000):
            a = model.predict(venv.normalize_obs(raw[None]), deterministic=True)[0][0]
            raw, _, term, _, info = env.step(a)
            if term: fell = True; break
            if t < 200: prev = dict(info['loaded']); continue
            ld = info['loaded']
            for f in ('left', 'right'):
                z = env.data.xpos[env.foot_body[f]][2]
                if ld[f]:
                    if not prev[f]:
                        td += 1
                        if fz[f] and f in stance_z: clear.append(max(fz[f])-stance_z[f])
                        fz[f] = []
                    stance_z[f] = z
                else:
                    fz[f].append(z)
            prev = dict(ld)
            s['vx'].append(info['v_body'][0]); s['flight'].append(float(info['flight']))
            s['hip'].append(env.data.xpos[env.base][2]); s['power'].append(float(np.sum(np.abs(info['power']))))
            s['act'].append(float(np.abs(a).mean()))
        M['fell'].append(fell)
        if fell: continue
        T = len(s['vx'])*.02; v = float(np.mean(s['vx'])); hz = td/T
        for k in ('vx', 'flight', 'hip', 'power', 'act'): M[k].append(float(np.mean(s[k])))
        M['step_hz'].append(hz); M['stride'].append(v/hz if hz else np.nan); M['clear'].append(float(np.median(clear)) if clear else np.nan)
    print(json.dumps(dict(run=d.name, cmd=vx, n=n, falls=int(sum(M['fell'])),
          **{k: round(float(np.nanmean(M[k])), 3) for k in M if k != 'fell'})), flush=True)
