"""Deterministic evaluation of a RecoverEnv (T1 get-up) policy: success rate and time to stand per fall mode.
Usage: .venv-sim/bin/python sim/rl/evaluate_recover.py sim/rl/runs/<name>/model.zip --episodes 10 [--video out.mp4]
Every number is a T1 (virtual actuator) result.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from evaluate import load, write_mp4  # noqa: E402
from recover_env import RecoverEnv, FALL_MODES  # noqa: E402


def episode(model, venv, env, mode, seconds, frames=None):
    venv.reset()
    obs = venv.normalize_obs(env.reset(options={'mode': mode})[0][None])
    t_stand, max_h = None, 0.
    for _ in range(int(seconds/.02)):
        action, _ = model.predict(obs, deterministic=True)
        raw, _, term, trunc, info = env.step(action[0])
        obs = venv.normalize_obs(raw[None])
        max_h = max(max_h, info['height'])
        if frames is not None:
            frames.append(env.render())
        if info['success']:
            t_stand = round(env.data.time-env.stand_hold*.02, 2)
            break
        if term:
            break
    return {'mode': mode, 'success': t_stand is not None, 'time_s': t_stand, 'max_height_m': round(max_h, 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--modes', nargs='+', default=list(FALL_MODES))
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seconds', type=float, default=8.)
    p.add_argument('--tail', choices=['active', 'locked'], default='active')
    p.add_argument('--video')
    p.add_argument('--out')
    a = p.parse_args()
    rows = []
    for mode in a.modes:
        eps = []
        for ep in range(a.episodes):
            env = RecoverEnv(randomize=False, seed=5000+ep, tail=a.tail, episode_s=a.seconds)
            model, venv = load(a.model, env)
            eps.append(episode(model, venv, env, mode, a.seconds))
        ok = [e['time_s'] for e in eps if e['success']]
        row = {'mode': mode, 'success': len(ok), 'episodes': len(eps), 'time_s': round(float(np.mean(ok)), 2) if ok else None,
               'max_height_m': round(float(np.mean([e['max_height_m'] for e in eps])), 3)}
        rows.append(row); print(json.dumps(row), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1)+'\n')
    if a.video:
        frames = []
        for mode in a.modes:
            env = RecoverEnv(randomize=False, seed=7, tail=a.tail, episode_s=a.seconds, render_mode='rgb_array')
            model, venv = load(a.model, env)
            episode(model, venv, env, mode, min(a.seconds, 6.), frames)
            env.close()
        write_mp4(a.video, frames)


if __name__ == '__main__':
    main()
