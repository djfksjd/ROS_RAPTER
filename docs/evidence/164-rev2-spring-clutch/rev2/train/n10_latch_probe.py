import json, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl'))
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
RUN = ROOT/'sim/rl/runs/rev2_n10'
kw = env_kwargs(json.loads((RUN/'args.json').read_text()))
out = {}
for cmd in (2., 4., 7.):
    env = RunEnv(**{**kw, 'episode_s': 30., 'level': 0.}, randomize=False, seed=7); env.init_seed = 96001
    model, venv = load(str(RUN/'model.zip'), env); venv.reset(); venv.reset()
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None]); d = env.data
    kq = [env.q_adr[env.ix[f'{s}_knee_pitch_joint']] for s in ('left', 'right')]
    rows = []
    for k in range(1000):
        a = model.predict(obs, deterministic=True)[0][0]
        raw, _, term, _, info = env.step(a); obs = venv.normalize_obs(raw[None])
        rows.append([d.qpos[kq[0]], d.qpos[kq[1]], info['loaded']['left'], info['loaded']['right'], env.latch_armed.get('left', False), env.latch_armed.get('right', False)])
        if term: break
    r = np.array(rows[250:], float); res = {}
    for j, s in enumerate(('left', 'right')):
        q, ld = r[:, j], r[:, 2+j] > 0
        sw, st = q[~ld], q[ld]
        # touchdown / lift-off knee angles
        td = [q[i] for i in range(1, len(q)) if ld[i] and not ld[i-1]]; lo = [q[i] for i in range(1, len(q)) if not ld[i] and ld[i-1]]
        res[s] = dict(duty=float(ld.mean()), swing_min=float(sw.min()), swing_max=float(sw.max()), stance_min=float(st.min()), stance_max=float(st.max()), stance_mean=float(st.mean()),
                      touchdown_mean=float(np.mean(td)), liftoff_mean=float(np.mean(lo)), n_td=len(td),
                      swing_below_q0_frac=float((sw < .956).mean()), stance_below_q0_frac=float((st < .956).mean()), armed_frac=float(r[:, 4+j].mean()))
    out[cmd] = res; print(cmd, json.dumps(res['left'], default=float)); print(cmd, json.dumps(res['right'], default=float))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
