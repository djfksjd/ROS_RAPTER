"""Deterministic evaluation of a trained policy per terrain and command; optional MP4.

Metrics per case (N episodes): fall rate, mean forward speed vs command, velocity tracking error,
distance travelled, max tilt. High reward alone is not reported as walking success.
Usage: .venv-sim/bin/python sim/rl/evaluate.py sim/rl/runs/<name>/model.zip [--video out.mp4]
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from raptor_env import RaptorEnv  # noqa: E402


def load(model_path, env):
    venv = VecNormalize.load(str(Path(model_path).with_name('vecnorm.pkl')), DummyVecEnv([lambda: env]))
    venv.training, venv.norm_reward = False, False
    return PPO.load(model_path, device='cpu'), venv


def episode(model, venv, env, command, seconds, frames=None):
    obs = venv.reset()
    env.command = np.array(command, float)
    env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    speeds, errs, tilt, fell = [], [], 0., False
    x0 = env.data.xpos[env.base][0]
    yaw = lambda: float(np.arctan2(env.data.xmat[env.base][3], env.data.xmat[env.base][0]))
    yaw0, yaw_rates = yaw(), []
    for _ in range(int(seconds/.02)):
        action, _ = model.predict(obs, deterministic=True)
        raw, _, term, trunc, info = env.step(action[0])
        obs = venv.normalize_obs(raw[None])
        yaw_rates.append(env.body_velocity(env.base)[0][2])
        speeds.append(info['v_body'][0]); errs.append(np.linalg.norm(info['v_body'][:2]-env.command[:2]))
        tilt = max(tilt, info['tilt'])
        if frames is not None:
            frames.append(env.render())
        if term:
            fell = True
            break
    return {'fell': fell, 'time_s': round(len(speeds)*.02, 2), 'mean_vx': round(float(np.mean(speeds[50:] or [0])), 3),
            'track_err': round(float(np.mean(errs[50:] or [0])), 3),
            'distance_m': round(float(env.data.xpos[env.base][0]-x0), 2), 'max_tilt': round(float(tilt), 3),
            'mean_yaw_rate': round(float(np.mean(yaw_rates[50:] or [0])), 3),
            'heading_change_rad': round(float(np.angle(np.exp(1j*(yaw()-yaw0)))), 2)}


def write_mp4(path, frames, fps=50):
    import subprocess
    h, w, _ = frames[0].shape
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                             '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                             '-crf', '24', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--terrain', nargs='+', default=['flat'])
    p.add_argument('--levels', type=float, nargs='+', default=[0.])
    p.add_argument('--commands', type=float, nargs='+', default=[0., .2, .4])
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--vel-scale', type=float, default=1.)
    p.add_argument('--actuator', default='urdf')
    p.add_argument('--dof', type=int, choices=[10, 12], default=10)
    p.add_argument('--sole', choices=['flat', 'rocker'], default='flat')
    p.add_argument('--jtc', type=float, default=0.)
    p.add_argument('--slew', type=float)
    p.add_argument('--dr', type=int, default=1)
    p.add_argument('--video')
    p.add_argument('--out')
    a = p.parse_args()
    rows = []
    for kind in a.terrain:
        for level in a.levels:
            for vx in a.commands:
                eps = []
                for ep in range(a.episodes):
                    env = RaptorEnv(kind, level=level, vel_scale=a.vel_scale, randomize=False, seed=1000+ep,
                                    actuator=a.actuator, dof=a.dof, sole=a.sole, jtc_horizon=a.jtc, slew=a.slew, dr=a.dr)
                    model, venv = load(a.model, env)
                    eps.append(episode(model, venv, env, [vx, 0., 0.], a.seconds))
                row = {'terrain': kind, 'level': level, 'cmd_vx': vx, 'falls': sum(e['fell'] for e in eps),
                       'episodes': len(eps), 'mean_vx': round(float(np.mean([e['mean_vx'] for e in eps])), 3),
                       'track_err': round(float(np.mean([e['track_err'] for e in eps])), 3),
                       'distance_m': round(float(np.mean([e['distance_m'] for e in eps])), 2),
                       'max_tilt': max(e['max_tilt'] for e in eps),
                       'mean_yaw_rate': round(float(np.mean([e['mean_yaw_rate'] for e in eps])), 3)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1)+'\n')
    if a.video:
        frames = []
        for kind in a.terrain:
            env = RaptorEnv(kind, level=a.levels[-1], vel_scale=a.vel_scale, randomize=False, seed=7, render_mode='rgb_array',
                            actuator=a.actuator, dof=a.dof, sole=a.sole, jtc_horizon=a.jtc, slew=a.slew, dr=a.dr)
            model, venv = load(a.model, env)
            episode(model, venv, env, [a.commands[-1], 0., 0.], min(a.seconds, 10.), frames)
            env.close()
        write_mp4(a.video, frames)


if __name__ == '__main__':
    main()
