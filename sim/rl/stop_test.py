"""Stopping authority and gait posture: walk at V for 8 s, then command 0 for 6 s (evidence 80).

Reports stop time (speed |v| < 0.1 m/s held 0.5 s), distance after the stop command, speed spread
after stopping, mean body pitch and CoM-minus-stance-foot x while walking. Deterministic, no randomization.
Usage: .venv-sim/bin/python sim/rl/stop_test.py MODEL.zip --dof 12 --sole rocker [--speed 0.6]
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from evaluate import load  # noqa: E402
from raptor_env import RaptorEnv  # noqa: E402


def stop_test(model_path, dof=12, sole='flat', speed=.6, walk_s=8., stop_s=6., jtc=0., slew=None):
    env = RaptorEnv('flat', dof=dof, sole=sole, randomize=False, seed=300, jtc_horizon=jtc, slew=slew)
    model, venv = load(model_path, env)
    venv.reset()
    env.resample_steps = 0
    offsets, pitch, after, stop_t, x_stop = [], [], [], None, None
    for k in range(int((walk_s+stop_s)/.02)):
        t = k*.02
        env.command = np.array([speed if t < walk_s else 0., 0, 0])
        action, _ = model.predict(venv.normalize_obs(env._obs()[None]), deterministic=True)
        _, _, fell, _, info = env.step(action[0])
        if fell:
            return {'fell_at_s': round(t, 2)}
        d = env.data
        if walk_s/2 < t < walk_s:
            feet, _ = env._contacts()
            loaded = [s for s in feet if feet[s] > 50]
            if loaded:
                offsets.append(d.subtree_com[env.base][0]-np.mean([d.xpos[env.foot_body[s]][0] for s in loaded]))
            R = d.xmat[env.base].reshape(3, 3)
            pitch.append(np.arcsin(-R[2, 0]))
        if t >= walk_s:
            if x_stop is None:
                x_stop = d.xpos[env.base][0]
            after.append(info['v_body'][0])
            if stop_t is None and len(after) >= 25 and np.all(np.abs(after[-25:]) < .1):
                stop_t = t-walk_s-.5
    return {'stop_time_s': None if stop_t is None else round(stop_t, 2),
            'distance_after_stop_m': round(float(env.data.xpos[env.base][0]-x_stop), 2),
            'speed_std_last_4s': round(float(np.std(after[-200:])), 3),
            'walking_pitch_rad': round(float(np.mean(pitch)), 3),
            'com_minus_stance_foot_x_m': round(float(np.mean(offsets)), 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--dof', type=int, default=12)
    p.add_argument('--sole', default='flat')
    p.add_argument('--speed', type=float, default=.6)
    p.add_argument('--jtc', type=float, default=0.)
    p.add_argument('--slew', type=float)
    a = p.parse_args()
    print(json.dumps(stop_test(a.model, a.dof, a.sole, a.speed, jtc=a.jtc, slew=a.slew)))


if __name__ == '__main__':
    main()
