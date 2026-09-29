"""Deterministic evaluation of a RunEnv (T1) policy: running metrics, tail active vs locked, impulse recovery.

Per command and episode: fall, mean speed, stride frequency, duty factor, flight fraction, peak GRF (BW) mean/std,
positive mechanical power and cost of transport, tail joint speed RMS, whole-body angular momentum RMS about the
CoM, tail-leg yaw momentum correlation (negative = the tail cancels the legs' yaw momentum). --impulses applies yaw
torque impulses (N·m·s, world z) at --impulse-at s and reports the recovery time (yaw rate and tilt back under
0.3 rad/s / 0.2 rad for 0.3 s) or the fall. --tail both runs the same policy with the tail free and held.
Every number is a T1 (virtual actuator) result; report it with run_env.ACTUATOR_T1.
Usage: .venv-sim/bin/python sim/rl/evaluate_run.py sim/rl/runs/<name>/model.zip --commands 2 5 8 11.1 --tail both
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from evaluate import load, write_mp4  # noqa: E402
from run_env import RunEnv  # noqa: E402


def episode(model, venv, env, command, seconds, lock_tail=False, impulse=None, frames=None):
    """impulse: (axis, N·m·s, time_s)."""
    obs = venv.reset()
    env.command = np.array(command, float)
    env.resample_steps = 0
    env.lock_tail = lock_tail or env.tail == 'locked'
    obs = venv.normalize_obs(env._obs()[None])
    log = {k: [] for k in ('vx', 'tilt', 'yaw_rate', 'power', 'flight', 'loaded', 'L', 'L_tail', 'L_legs', 'tail_qd')}
    fell, applied, recovered_at, settle = False, False, None, 0
    x0 = env.data.xpos[env.base][0]
    for _ in range(int(seconds/.02)):
        if impulse and not applied and env.data.time >= impulse[2]:
            env.apply_impulse(impulse[0], impulse[1])
            applied = True
        action, _ = model.predict(obs, deterministic=True)
        raw, _, term, trunc, info = env.step(action[0])
        obs = venv.normalize_obs(raw[None])
        w = env.body_velocity(env.base)[0]
        log['vx'].append(info['v_body'][0]); log['tilt'].append(info['tilt']); log['yaw_rate'].append(w[2])
        log['power'].append(info['power']); log['flight'].append(info['flight'])
        log['loaded'].append(sum(info['loaded'].values()))
        for k in ('L', 'L_tail', 'L_legs', 'tail_qd'):
            log[k].append(info[k])
        if applied and recovered_at is None:
            settle = settle+1 if abs(w[2]) < .3 and info['tilt'] < .2 else 0
            if settle >= 15:
                recovered_at = round(env.data.time-.3-impulse[2], 2)
        if frames is not None:
            frames.append(env.render())
        if term:
            fell = True
            break
    n = len(log['vx']); skip = min(50, n//2)
    vx = np.array(log['vx'][skip:] or [0.]); power = np.array(log['power'][skip:] or [0.])
    td = [t for t, _ in env.touchdowns if t > skip*.02]
    L = np.array(log['L'][skip:] or [[0, 0, 0]]); Lt = np.array(log['L_tail'][skip:] or [[0, 0, 0]]); Ll = np.array(log['L_legs'][skip:] or [[0, 0, 0]])
    corr = float(np.corrcoef(Lt[:, 2], Ll[:, 2])[0, 1]) if len(Lt) > 10 and Lt[:, 2].std() > 1e-6 and Ll[:, 2].std() > 1e-6 else 0.
    W = env.weight
    return {'fell': fell, 'time_s': round(n*.02, 2), 'mean_vx': round(float(vx.mean()), 2),
            'distance_m': round(float(env.data.xpos[env.base][0]-x0), 1),
            'stride_hz': round(len(td)/2/max((n-skip)*.02, 1e-3), 2),
            'duty': round(float(np.mean(np.array(log['loaded'][skip:] or [0]) > 0)), 2),
            'flight_frac': round(float(np.mean(log['flight'][skip:] or [0])), 2),
            'peak_grf_bw': round(float(np.mean(env.peaks)), 2) if env.peaks else 0.,
            'peak_grf_sd': round(float(np.std(env.peaks)), 2) if env.peaks else 0.,
            'power_w': round(float(power.mean()), 1),
            'cot': round(float(power.mean()/(W*max(abs(vx.mean()), .1))), 2),
            'tail_qd_rms': round(float(np.sqrt(np.mean(np.square(log['tail_qd'][skip:] or [[0, 0]])))), 2),
            'L_yz_rms': round(float(np.sqrt(np.mean(L[:, 1]**2+L[:, 2]**2))), 3),
            'tail_leg_yaw_corr': round(corr, 2),
            'max_tilt': round(float(max(log['tilt'])), 2),
            'recovery_s': recovered_at, 'impulse': impulse[1] if impulse else None}


def mean_row(eps, keys):
    return {k: round(float(np.mean([e[k] for e in eps])), 2) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--commands', type=float, nargs='+', default=[0., 2., 5., 8., 11.1])
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--mass', type=float, default=5.)
    p.add_argument('--springs', choices=['none', 'b', 'c'], default='c')
    p.add_argument('--tail', choices=['active', 'locked', 'both'], default='active')
    p.add_argument('--impulses', type=float, nargs='+', help='yaw impulses N·m·s (world z) applied at --impulse-at')
    p.add_argument('--impulse-axis', choices=['y', 'z'], default='z')
    p.add_argument('--impulse-at', type=float, default=4.)
    p.add_argument('--impulse-cmd', type=float, default=5.)
    p.add_argument('--no-obs-vel', action='store_true')
    p.add_argument('--video')
    p.add_argument('--out')
    a = p.parse_args()
    modes = ['active', 'locked'] if a.tail == 'both' else [a.tail]
    keys = ('mean_vx', 'distance_m', 'stride_hz', 'duty', 'flight_frac', 'peak_grf_bw', 'peak_grf_sd', 'power_w', 'cot',
            'tail_qd_rms', 'L_yz_rms', 'tail_leg_yaw_corr', 'max_tilt')
    rows = []

    def new_env(seed, render=False):
        return RunEnv(mass=a.mass, springs=a.springs, randomize=False, seed=seed, obs_vel=not a.no_obs_vel,
                      render_mode='rgb_array' if render else None)
    for mode in modes:
        for vx in a.commands:
            eps = []
            for ep in range(a.episodes):
                env = new_env(1000+ep)
                model, venv = load(a.model, env)
                eps.append(episode(model, venv, env, [vx, 0., 0.], a.seconds, lock_tail=mode == 'locked'))
            row = {'tail': mode, 'cmd_vx': vx, 'falls': sum(e['fell'] for e in eps), 'episodes': len(eps), **mean_row(eps, keys)}
            rows.append(row); print(json.dumps(row), flush=True)
        for J in a.impulses or []:
            eps = []
            for ep in range(a.episodes):
                env = new_env(2000+ep)
                model, venv = load(a.model, env)
                eps.append(episode(model, venv, env, [a.impulse_cmd, 0., 0.], a.seconds, lock_tail=mode == 'locked',
                                   impulse=(a.impulse_axis, J, a.impulse_at)))
            rec = [e['recovery_s'] for e in eps if e['recovery_s'] is not None and not e['fell']]
            row = {'tail': mode, 'impulse_axis': a.impulse_axis, 'impulse_nms': J, 'cmd_vx': a.impulse_cmd,
                   'falls': sum(e['fell'] for e in eps), 'episodes': len(eps),
                   'recovered': len(rec), 'recovery_s': round(float(np.mean(rec)), 2) if rec else None,
                   'max_tilt': round(float(np.mean([e['max_tilt'] for e in eps])), 2)}
            rows.append(row); print(json.dumps(row), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1)+'\n')
    if a.video:
        frames = []
        env = new_env(7, render=True)
        model, venv = load(a.model, env)
        episode(model, venv, env, [a.commands[-1], 0., 0.], min(a.seconds, 10.), frames=frames)
        env.close()
        write_mp4(a.video, frames)


if __name__ == '__main__':
    main()
