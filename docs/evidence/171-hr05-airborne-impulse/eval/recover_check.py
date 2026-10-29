"""Is a counted fall final? Same setting as roll_diag (only roll impulses, rest randomized, 65 s), but the episode is not
stopped at the fall condition (tilt > 0.8 rad, body contact or hip below half height): the run continues 2 s and records
whether the robot comes back (tilt < 0.3 rad and hip above 0.8 of nominal for the last 0.5 s, no body contact).
argv: run_dir cmd episodes."""
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
for ep in range(n):
    env = RunEnv(**kw, randomize=True, seed=5000+ep); env.init_seed = None
    model, venv = load(str(d/'model.zip'), env); venv.reset()
    env.command = np.array([vx, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    t_fall, rec = None, []
    for t in range(3000):
        raw, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0][0])
        obs = venv.normalize_obs(raw[None])
        if t_fall is None and term:
            t_fall = env.data.time
        if t_fall is not None:
            rec.append((info['tilt'], env.data.qpos[2]/env.h_nom))
            if env.data.time-t_fall >= 2.:
                break
    if t_fall is None:
        print(json.dumps(dict(seed=5000+ep, cmd=vx, fell=False))); continue
    r = np.array(rec); last = r[-25:]
    print(json.dumps(dict(seed=5000+ep, cmd=vx, fell=True, t_fall=round(t_fall, 2), tilt_max=round(float(r[:, 0].max()), 2),
                          h_min=round(float(r[:, 1].min()), 2), recovered=bool((last[:, 0] < .3).all() and (last[:, 1] > .8).all()))))
