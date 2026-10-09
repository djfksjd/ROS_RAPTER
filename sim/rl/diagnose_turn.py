"""Read-only T1 turn diagnosis: reconstruct a saved run, retain per-step evidence."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs


def diagnose(run_dir, yaw_cmd, seed, speed=4., seconds=10., turn_at=4., turn_seconds=2.):
    args = json.loads((Path(run_dir)/'args.json').read_text())
    env = RunEnv(**{**env_kwargs(args), 'episode_s': seconds+5., 'level': 0.},
                 randomize=False, seed=7)
    env.init_seed = seed
    model, venv = load(str(Path(run_dir)/'model.zip'), env)
    venv.reset()
    env.command = np.array([speed, 0., 0.]); env.resample_steps = 0
    rows = []
    for _ in range(int(seconds/.02)):
        turning = turn_at <= round(float(env.data.time), 12) < turn_at+turn_seconds
        env.command[2] = yaw_cmd if turning else 0.
        obs = venv.normalize_obs(env._obs()[None])
        action, _ = model.predict(obs, deterministic=True)
        _, reward, term, _, info = env.step(action[0])
        rows.append(dict(t=float(env.data.time), turning=turning,
                         command=env.command.tolist(), vx=float(info['v_body'][0]),
                         yaw_rate=float(env.body_velocity(env.base)[0][2]), tilt=float(info['tilt']),
                         terms={k: float(v) for k, v in info['terms'].items()},
                         reward=float(reward), heat=env.heat_inst.tolist(), action=action[0].tolist()))
        if term:
            break
    turn = [r for r in rows if r['turning']]
    mean = lambda key: float(np.mean([r[key] for r in turn])) if turn else None
    result = dict(yaw_cmd=yaw_cmd, seed=seed, speed_cmd=speed,
                  turn_at_s=turn_at, turn_duration_s=turn_seconds,
                  turn_samples=len(turn),
                  fell=bool(term), time_s=rows[-1]['t'],
                  turn_yaw_mean=mean('yaw_rate'), turn_vx_mean=mean('vx'),
                  weighted_terms={k: env.weights[k]*float(np.mean([r['terms'][k] for r in turn])) if turn else None
                                  for k in env.weights},
                  active_joints=env.active, rows=rows)
    venv.close()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--out', required=True)
    p.add_argument('--yaw', type=float, nargs='+', default=[-1., -.5, 0., .5, 1.])
    p.add_argument('--seeds', type=int, nargs='+', default=[5001, 5002])
    p.add_argument('--speed', type=float, default=4.)
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--turn-at', type=float, default=4.)
    p.add_argument('--turn-seconds', type=float, default=2.)
    a = p.parse_args()
    if a.turn_at < 0 or a.turn_seconds <= 0 or a.seconds < a.turn_at+a.turn_seconds:
        p.error('turn interval must be positive and contained in the evaluation duration')
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        p.error(f'refusing to overwrite {out}')
    with out.open('x') as f:
        for yaw in a.yaw:
            for seed in a.seeds:
                row = diagnose(a.run_dir, yaw, seed, a.speed, a.seconds,
                               a.turn_at, a.turn_seconds)
                f.write(json.dumps(row)+'\n'); f.flush()
                print(json.dumps({k: v for k, v in row.items() if k not in ('rows', 'active_joints')}), flush=True)


if __name__ == '__main__':
    main()
