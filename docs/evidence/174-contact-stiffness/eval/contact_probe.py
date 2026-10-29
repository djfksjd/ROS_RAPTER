"""Ground contact stiffness probe: a trained policy on flat ground with every contact's solref time constant set to tc
(MuJoCo default 0.02; Cassie in mujoco_menagerie uses 0.005 at 0.5 ms). No retraining, randomize off, realism from
args.json at nominal. argv: run_dir tc cmd episodes [seconds=20]. Logs falls, speed and the deepest ground penetration
per control step (median / p95 / max over all steps, metres)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'sim/rl')
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
d, tc, vx, n = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
secs = float(sys.argv[5]) if len(sys.argv) > 5 else 20.
_build = RunEnv._build
def build(self):
    _build(self)
    self.model.geom_solref[:, 0] = tc
RunEnv._build = build
kw = {**env_kwargs(json.loads((d/'args.json').read_text())), 'level': 0., 'episode_s': secs+5.}
rows, pen = [], []
for ep in range(n):
    env = RunEnv(**kw, randomize=False, seed=5000+ep); env.init_seed = 5000+ep
    model, venv = load(str(d/'model.zip'), env)
    venv.reset(); env.command = np.array([vx, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    m, dd = env.model, env.data
    fell, vxs = False, []
    for t in range(int(secs/.02)):
        raw, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0][0])
        obs = venv.normalize_obs(raw[None])
        if term: fell = True; break
        ds = [dd.contact[i].dist for i in range(dd.ncon) if dd.contact[i].geom1 in env.ground or dd.contact[i].geom2 in env.ground]
        if ds and t >= 50: pen.append(-min(ds))
        vxs.append(info['v_body'][0])
    rows.append(dict(fell=fell, vx=float(np.mean(vxs[100:])) if len(vxs) > 100 else np.nan))
p = np.array(pen) if pen else np.zeros(1)
print(json.dumps(dict(run=d.name, tc=tc, cmd=vx, n=n, falls=sum(r['fell'] for r in rows),
                      vx=round(float(np.nanmean([r['vx'] for r in rows])), 2),
                      pen_med=round(float(np.median(p)), 4), pen_p95=round(float(np.percentile(p, 95)), 4), pen_max=round(float(p.max()), 4))), flush=True)
