"""Read-only policy diagnosis: change yaw command at identical physical states.

Counterfactual predictions are not executed. The actual rollout uses straight
4 m/s motion. This separates command reception from closed-loop turn ability.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs


def probe(run_dir, seed):
    run = Path(run_dir)
    args = json.loads((run/'args.json').read_text())
    env = RunEnv(**{**env_kwargs(args), 'level': 0., 'episode_s': 12.},
                 randomize=False, seed=7)
    env.init_seed = seed
    model, venv = load(str(run/'model.zip'), env)
    rows = []
    try:
        venv.reset()
        env.command = np.array([4., 0., 0.]); env.resample_steps = 0
        fell = False
        for _ in range(400):
            predictions = {}
            normalized = {}
            targets = {}
            for yaw in [-1., 0., 1.] if env.data.time >= 4. else [0.]:
                env.command[2] = yaw
                obs = venv.normalize_obs(env._obs()[None])
                action, _ = model.predict(obs, deterministic=True)
                predictions[yaw] = action[0].copy()
                normalized[yaw] = obs[0, 6:9].tolist()
                ix = env.policy_idx
                targets[yaw] = np.clip(env.q0[ix]+env.scale[ix]*action[0],
                                       env.lo[ix], env.hi[ix]).tolist()
            if env.data.time >= 4.:
                rows.append(dict(time_s=float(env.data.time), phase=float(env.phase),
                                 normalized_commands=normalized,
                                 actions={k:v.tolist() for k,v in predictions.items()},
                                 proposed_target_rad=targets))
            env.command[2] = 0.
            _, _, fell, _, _ = env.step(predictions[0.])
            if fell:
                break
        delta = np.array([np.array(r['actions'][1.])-np.array(r['actions'][-1.]) for r in rows])
        return dict(run=str(run), seed=seed, fell=bool(fell), states=len(rows),
                    policy_joints=[env.active[i] for i in env.policy_idx],
                    mean_abs_action_difference= np.mean(np.abs(delta), axis=0).tolist() if len(rows) else None,
                    rms_action_difference=np.sqrt(np.mean(delta**2, axis=0)).tolist() if len(rows) else None,
                    rows=rows)
    finally:
        venv.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[8001, 8002])
    a = p.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        p.error(f'refusing to overwrite {out}')
    with out.open('x') as f:
        for run in a.runs:
            for seed in a.seeds:
                row = probe(run, seed)
                f.write(json.dumps(row)+'\n'); f.flush()
                print(json.dumps({k:v for k,v in row.items() if k!='rows'}), flush=True)


if __name__ == '__main__':
    main()
