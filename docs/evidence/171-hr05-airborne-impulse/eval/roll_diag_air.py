"""Evidence 171: evidence 170 roll_diag_signed plus contact at the impulse: air0 = no foot loaded at the impulse step,
air_frac = fraction of the next 0.3 s with no foot loaded (air_01 / air_02: first 0.1 / 0.2 s; tilt_01 / tilt_03: tilt at
0.1 / 0.3 s, to tell lost contact from tipping), stance_n = loaded feet at the impulse (env.loaded). Base: evidence 168 roll_diag plus signed hip-roll use: out_frac / in_frac = largest outward (abduction) / inward hip-roll angle
over both legs as a fraction of the joint limit (both joint axes are +x, so outward is +q left, -q right). Roll-impulse fall mechanism: for each roll impulse (only roll impulses on, rest randomized), the 1.5 s after it:
hip-roll action saturation (|a| >= .98 of the +-scale band), hip-roll angle vs its joint limit, hip-roll actuator force vs
its limit, body tilt; and whether the robot fell within 2.5 s. argv: run_dir cmd episodes."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'sim/rl')
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
d, vx, n = Path(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
kw = {**env_kwargs(json.loads((d/'args.json').read_text())), 'level': 0., 'episode_s': 65., 'yaw_impulse': None,
      'disturb_items': ['pitch', 'yaw', 'trip', 'touchdown']}
_imp = RunEnv.apply_impulse
def _log(self, axis, impulse, duration=.04, body=None):
    self.imp_log.append((float(self.data.time), float(impulse)))
    return _imp(self, axis, impulse, duration, body)
RunEnv.apply_impulse = _log
out = []
for ep in range(n):
    env = RunEnv(**kw, randomize=True, seed=5000+ep); env.init_seed = None
    model, venv = load(str(d/'model.zip'), env); venv.reset(); env.imp_log = []
    env.command = np.array([vx, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    hr = [env.active.index('left_hip_roll_joint'), env.active.index('right_hip_roll_joint')]
    pi = [list(env.policy_idx).index(i) for i in hr]
    rec, t_fall = [], None
    for t in range(3000):
        act = model.predict(obs, deterministic=True)[0][0]
        raw, _, term, _, info = env.step(act)
        obs = venv.normalize_obs(raw[None])
        m, dd = env.model, env.data
        q = dd.qpos[env.q_adr[hr]]; f = dd.actuator_force[env.act[hr]]; lim = np.abs(m.actuator_forcerange[env.act[hr], 1])
        rec.append((dd.time, np.abs(act[pi]).max(), np.max(np.abs(q)/env.hi[hr]), np.max(np.abs(f)/np.maximum(lim, 1e-6)), float(info['tilt']), max(q[0], -q[1])/env.hi[hr[0]], max(-q[0], q[1])/env.hi[hr[0]], float(sum(env.loaded.values()))))
        if term: t_fall = dd.time; break
    rec = np.array(rec)
    for ti, mag in env.imp_log:
        w = rec[(rec[:, 0] >= ti) & (rec[:, 0] < ti+1.5)]
        if not len(w): continue
        out.append(dict(seed=5000+ep, cmd=vx, mag=round(abs(mag), 2), fell=t_fall is not None and 0 <= t_fall-ti < 2.5,
                        act_sat=round(float((w[:, 1] >= .98).mean()), 3), q_frac=round(float(w[:, 2].max()), 3),
                        f_frac=round(float(w[:, 3].max()), 3), f_sat=round(float((w[:, 3] >= .98).mean()), 3), tilt_max=round(float(w[:, 4].max()), 3),
                        out_frac=round(float(w[:, 5].max()), 3), in_frac=round(float(w[:, 6].max()), 3),
                        stance_n=int(w[0, 7]), air0=bool(w[0, 7] == 0), air_frac=round(float((w[w[:, 0] < ti+.3][:, 7] == 0).mean()), 3),
                        air_01=round(float((w[w[:, 0] < ti+.1][:, 7] == 0).mean()), 3), air_02=round(float((w[w[:, 0] < ti+.2][:, 7] == 0).mean()), 3),
                        tilt_01=round(float(w[w[:, 0] < ti+.1][-1, 4]), 3), tilt_03=round(float(w[w[:, 0] < ti+.3][-1, 4]), 3),
                        t_fall_after=None if t_fall is None else round(float(t_fall-ti), 2)))
for r in out: print(json.dumps(r))
