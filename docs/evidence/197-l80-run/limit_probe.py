"""Evidence 197: which joint limits l80_r1. Constant forward command from standing, randomize off, realism on.
Per policy motor joint: share of steps at the drive torque limit (|actuator force| >= 0.98 of the force range of that
step), RMS joint torque, peak joint speed, hottest winding and lowest derate. Usage:
.venv-sim/bin/python docs/evidence/197-l80-run/limit_probe.py RUN_DIR CMD SECONDS"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
import stage_gates as sg  # noqa: E402

d, cmd, secs = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
a = json.loads((d/'args.json').read_text())
kw = {**sg.env_kwargs(a), 'level': 0., 'episode_s': secs+5., 'disturb_items': [], 'roll_impulse': None, 'yaw_impulse': None,
      'cmd_max': [0., 0., 0.], 'init_speed': False}
env = sg.RunEnv(**kw, randomize=False, seed=6800); env.init_seed = 6800
model, venv = sg.load(str(d/'model.zip'), env); venv.reset()
env.resample_steps = 10**9; env.command = np.array([cmd, 0., 0.])
obs = venv.normalize_obs(env._obs()[None]); m, dd = env.model, env.data
names = [n.replace('_joint', '') for n in env.active]
keep = [i for i, n in enumerate(names) if ('left' in n or 'tail' in n) and 'ankle' not in n]
sat, tau, w, v = [], [], [], []
tw, der = np.zeros(len(names)), np.ones(len(names))
fell = None
for k in range(int(secs/.02)):
    raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None])
    f = dd.actuator_force[env.act]
    lim = np.where(f >= 0, m.actuator_forcerange[env.act, 1], -m.actuator_forcerange[env.act, 0])
    sat.append((lim > 1e-6) & (np.abs(f) >= .98*lim))
    tau.append(np.abs(f)); w.append(np.abs(dd.qvel[env.v_adr]))
    R = dd.xmat[env.base].reshape(3, 3); v.append(float((R.T @ dd.qvel[:3])[0]))
    r = info.get('realism', {})
    if 't_winding' in r:
        tw = np.fmax(tw, np.nan_to_num(r['t_winding'])); der = np.fmin(der, r['derate'])
    if term:
        fell = round((k+1)*.02, 2); break
S, T, W = np.array(sat), np.array(tau), np.array(w)
out = dict(run=d.name, cmd=cmd, secs=secs, fell=fell, v_mean=round(float(np.mean(v[len(v)//2:])), 2), joints={
    names[i]: dict(at_limit=round(float(S[:, i].mean()), 3), rms_nm=round(float(np.sqrt((T[:, i]**2).mean())), 1),
                   peak_rad_s=round(float(W[:, i].max()), 1), winding_max_c=round(float(tw[i]), 1), derate_min=round(float(der[i]), 2))
    for i in keep})
print(json.dumps(out))
