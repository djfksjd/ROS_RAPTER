"""Mean reward terms and heat state at one command on flat stiff ground, nominal. argv: run_dir cmd episodes [seconds=30].
Shows which training reward terms are active at high speed (e.g. heat above heat_limit)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'sim/rl')
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs
d, vx, n = Path(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]); secs = float(sys.argv[4]) if len(sys.argv) > 4 else 30.
kw = {**env_kwargs(json.loads((d/'args.json').read_text())), 'level': 0., 'episode_s': secs+5.}
T, H, V = {}, [], []
for ep in range(n):
    env = RunEnv(**kw, randomize=False, seed=5000+ep); env.init_seed = 5000+ep
    model, venv = load(str(d/'model.zip'), env)
    venv.reset(); env.command = np.array([vx, 0., 0.]); env.resample_steps = 0
    raw = env._obs()
    for t in range(int(secs/.02)):
        raw, _, term, _, info = env.step(model.predict(venv.normalize_obs(raw[None]), deterministic=True)[0][0])
        if term: break
        if t >= 250:
            for k, x in info['terms'].items(): T.setdefault(k, []).append(float(x))
            H.append(info['heat']); V.append(info['v_body'][0])
H = np.array(H)
print(json.dumps(dict(run=d.name, cmd=vx, n=n, vx=round(float(np.mean(V)), 3), heat_limit=env.heat_limit,
      heat_mean=np.round(H.mean(0), 3).tolist(), heat_max=np.round(H.max(0), 3).tolist(),
      terms={k: round(float(np.mean(x)), 4) for k, x in T.items()})), flush=True)
