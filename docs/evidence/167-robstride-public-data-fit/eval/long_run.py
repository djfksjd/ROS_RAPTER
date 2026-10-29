"""Long continuous run, randomize off, realism v2: winding temperature over time. argv: run_dir cmd seconds mount
(mount 'plate' = fitted vendor heat-sink values, 'worst' = r_th_ha x2.0 and c_th_h x0.2, the ends of thermal.dr)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'sim/rl')
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
d, vx, secs, mount = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
env = RunEnv(**{**env_kwargs(json.loads((d/'args.json').read_text())), 'level': 0., 'episode_s': secs+5.}, randomize=False, seed=5000)
env.init_seed = 5000
model, venv = load(str(d/'model.zip'), env)
venv.reset()
if mount == 'worst':
    th = env.rz.thermal; th.r_ha = th.r_ha*2.0; th.c_h = th.c_h*0.2
env.command = np.array([vx, 0., 0.]); env.resample_steps = 0
obs = venv.normalize_obs(env._obs()[None])
trace, fell, der = [], None, 1.
for t in range(int(secs/.02)):
    raw, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0][0])
    obs = venv.normalize_obs(raw[None])
    if term: fell = round((t+1)*.02, 2); break
    r = info['realism']; der = min(der, float(np.min(r['derate'])))
    if (t+1) % 1500 == 0:  # every 30 s
        trace.append(dict(t=(t+1)*.02, vx=round(float(info['v_body'][0]), 2), t_winding=[round(float(x), 1) for x in r['t_winding']]))
print(json.dumps(dict(run=d.name, cmd=vx, secs=secs, mount=mount, fell=fell, min_derate=round(der, 3), motors=env.rz.motor_name, trace=trace)), flush=True)
