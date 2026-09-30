"""Deterministic evaluation of a RunEnv (T1) policy: running metrics, tail active vs locked, impulse recovery.

Per command and episode: fall, mean speed, path length, heading drift (mean yaw rate), stride frequency, duty factor, flight fraction, peak GRF (BW) mean/std,
positive mechanical power and cost of transport, tail joint speed RMS, whole-body angular momentum RMS about the
CoM, tail-leg yaw momentum correlation (negative = the tail cancels the legs' yaw momentum). --impulses applies yaw
torque impulses (N·m·s, world z) at --impulse-at s and reports the recovery time; --turns applies
yaw-rate command steps (fast direction changes) while running and reports the heading change achieved, the minimum
speed and the maximum tilt during the turn, (yaw rate and tilt back under
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


def episode(model, venv, env, command, seconds, lock_tail=False, impulse=None, frames=None, turn=None):
    """impulse: (axis, N·m·s, time_s). turn: (yaw_rate_cmd rad/s, start_s, duration_s): a heading-rate step while running."""
    obs = venv.reset()
    env.command = np.array(command, float)
    env.resample_steps = 0
    env.lock_tail = lock_tail or env.tail == 'locked'
    obs = venv.normalize_obs(env._obs()[None])
    log = {k: [] for k in ('vx', 'tilt', 'yaw_rate', 'power', 'flight', 'loaded', 'left', 'right', 'L', 'L_tail', 'L_legs', 'tail_qd')}
    fell, applied, recovered_at, settle = False, False, None, 0
    base_w, base_tilt = [], []
    yaw = lambda: float(np.arctan2(env.data.xmat[env.base][3], env.data.xmat[env.base][0]))
    turn_log = {'yaw0': None, 'yaw1': None, 'min_vx': None, 'max_tilt': 0., 'acc': 0., 'prev': None}
    path, prev = 0., env.data.xpos[env.base][:2].copy()
    for _ in range(int(seconds/.02)):
        if impulse and not applied and env.data.time >= impulse[2]:
            env.apply_impulse(impulse[0], impulse[1])
            applied = True
        if turn:
            t = env.data.time
            if t >= turn[1] and turn_log['yaw0'] is None:
                turn_log['yaw0'] = yaw()
            in_turn = turn[1] <= t < turn[1]+turn[2]
            env.command[2] = turn[0] if in_turn else 0.
            if turn_log['yaw0'] is not None and turn_log['yaw1'] is None and t >= turn[1]+turn[2]:
                turn_log['yaw1'] = yaw()
        action, _ = model.predict(obs, deterministic=True)
        raw, _, term, trunc, info = env.step(action[0])
        if turn and turn_log['yaw0'] is not None and turn_log['yaw1'] is None:  # unwrapped heading change
            y = yaw()
            if turn_log['prev'] is not None:
                turn_log['acc'] += float(np.angle(np.exp(1j*(y-turn_log['prev']))))
            turn_log['prev'] = y
        if turn and turn[1] <= env.data.time < turn[1]+turn[2]+1.:
            v = info['v_body'][0]
            turn_log['min_vx'] = v if turn_log['min_vx'] is None else min(turn_log['min_vx'], v)
            turn_log['max_tilt'] = max(turn_log['max_tilt'], info['tilt'])
        obs = venv.normalize_obs(raw[None])
        w = env.body_velocity(env.base)[0]
        pos = env.data.xpos[env.base][:2].copy(); path += float(np.linalg.norm(pos-prev)); prev = pos
        log['vx'].append(info['v_body'][0]); log['tilt'].append(info['tilt']); log['yaw_rate'].append(w[2])
        log['power'].append(info['power']); log['flight'].append(info['flight'])
        log['loaded'].append(sum(info['loaded'].values()))
        log['left'].append(bool(info['loaded']['left'])); log['right'].append(bool(info['loaded']['right']))
        for k in ('L', 'L_tail', 'L_legs', 'tail_qd'):
            log[k].append(info[k])
        if impulse and not applied:
            base_w.append(abs(w[2])); base_tilt.append(info['tilt'])
        if applied and recovered_at is None:
            # recovered: yaw rate and tilt back inside the pre-impulse band (95th percentile of the last 1 s, at least
            # 0.3 rad/s / 0.2 rad) for 0.3 s. A running tail keeps the yaw rate oscillating, so a fixed 0.3 rad/s
            # threshold was never met by the tail-active policy (evidence 86).
            thr_w = max(.3, 1.2*float(np.percentile(base_w[-50:], 95))) if base_w else .3
            thr_t = max(.2, 1.2*float(np.percentile(base_tilt[-50:], 95))) if base_tilt else .2
            settle = settle+1 if abs(w[2]) < thr_w and info['tilt'] < thr_t else 0
            if settle >= 15:
                recovered_at = round(env.data.time-.3-impulse[2], 2)
        if frames is not None:
            frames.append(env.render())
        if term:
            fell = True
            break
    n = len(log['vx']); skip = min(50, n//2)
    vx = np.array(log['vx'][skip:] or [0.]); power = np.array(log['power'][skip:] or [0.])
    # touchdowns from debounced per-foot contact: a stance counts only if it lasts >= 3 control steps (60 ms);
    # brief flight-phase touches otherwise inflate the stride count and scramble the phase metric
    def stances(flags):
        out, i = [], skip
        while i < n:
            if flags[i]:
                j = i
                while j < n and flags[j]:
                    j += 1
                if j-i >= 3:
                    out.append(i*.02)
                i = j
            else:
                i += 1
        return out
    tl, tr = stances(log['left']), stances(log['right'])
    td = sorted(tl+tr)
    # gait symmetry: phase of each right touchdown after the preceding left one, in left-stride units
    # (0.5 = alternating run, ~0 or ~1 = bound); per-leg duty = fraction of time each foot is loaded
    lr = []
    if len(tl) > 2:
        T_l = float(np.mean(np.diff(tl)))
        for t in tr:
            prev = [x for x in tl if x <= t]
            if prev:
                lr.append(((t-prev[-1])/T_l) % 1.)
    lr_phase = float(np.mean(lr)) if lr else None
    leg_duty = float(np.mean(np.array(log['loaded'][skip:] or [0])))/2 if n > skip else 0.
    L = np.array(log['L'][skip:] or [[0, 0, 0]]); Lt = np.array(log['L_tail'][skip:] or [[0, 0, 0]]); Ll = np.array(log['L_legs'][skip:] or [[0, 0, 0]])
    corr = float(np.corrcoef(Lt[:, 2], Ll[:, 2])[0, 1]) if len(Lt) > 10 and Lt[:, 2].std() > 1e-6 and Ll[:, 2].std() > 1e-6 else 0.
    W = env.weight
    return {'fell': fell, 'time_s': round(n*.02, 2), 'mean_vx': round(float(vx.mean()), 2),
            'path_m': round(path, 1), 'yaw_drift': round(float(np.mean(log['yaw_rate'][skip:] or [0])), 2),
            'stride_hz': round(len(tl)/max((n-skip)*.02, 1e-3), 2),
            'duty': round(float(np.mean(np.array(log['loaded'][skip:] or [0]) > 0)), 2),
            'leg_duty': round(leg_duty, 2), 'lr_phase': round(lr_phase, 2) if lr_phase is not None else None,
            'flight_frac': round(float(np.mean(log['flight'][skip:] or [0])), 2),
            'peak_grf_bw': round(float(np.mean(env.peaks)), 2) if env.peaks else 0.,
            'peak_grf_sd': round(float(np.std(env.peaks)), 2) if env.peaks else 0.,
            'power_w': round(float(power.mean()), 1),
            'cot': round(float(power.mean()/(W*max(abs(vx.mean()), .1))), 2),
            'tail_qd_rms': round(float(np.sqrt(np.mean(np.square(log['tail_qd'][skip:] or [[0, 0]])))), 2),
            'L_yz_rms': round(float(np.sqrt(np.mean(L[:, 1]**2+L[:, 2]**2))), 3),
            'tail_leg_yaw_corr': round(corr, 2),
            'max_tilt': round(float(max(log['tilt'])), 2),
            'recovery_s': recovered_at, 'impulse': impulse[1] if impulse else None,
            'turn_rad': round(turn_log['acc'], 2) if turn and turn_log['yaw1'] is not None else None,
            'turn_min_vx': round(float(turn_log['min_vx']), 2) if turn and turn_log['min_vx'] is not None else None,
            'turn_max_tilt': round(float(turn_log['max_tilt']), 2) if turn else None}


def mean_row(eps, keys):
    out = {}
    for k in keys:  # None (e.g. lr_phase when an episode fell before two strides) is skipped
        vals = [e[k] for e in eps if e[k] is not None]
        out[k] = round(float(np.mean(vals)), 2) if vals else None
    return out


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
    p.add_argument('--turns', type=float, nargs='+', help='yaw-rate command steps rad/s applied at --turn-at for --turn-s')
    p.add_argument('--turn-at', type=float, default=4.)
    p.add_argument('--turn-s', type=float, default=2.)
    p.add_argument('--turn-cmd', type=float, nargs='+', default=[4.], help='forward speeds for the turn test')
    p.add_argument('--no-obs-vel', action='store_true')
    p.add_argument('--ankle-clutch', action='store_true')
    p.add_argument('--couple-ankle', action='store_true')
    p.add_argument('--video')
    p.add_argument('--out')
    a = p.parse_args()
    modes = ['active', 'locked'] if a.tail == 'both' else [a.tail]
    keys = ('mean_vx', 'path_m', 'yaw_drift', 'stride_hz', 'duty', 'leg_duty', 'lr_phase', 'flight_frac', 'peak_grf_bw', 'peak_grf_sd', 'power_w', 'cot',
            'tail_qd_rms', 'L_yz_rms', 'tail_leg_yaw_corr', 'max_tilt')
    rows = []

    def new_env(seed, render=False):
        # a policy trained with --tail locked has 10 actions; 'both' evaluates a tail-active policy with the tail held
        return RunEnv(mass=a.mass, springs=a.springs, randomize=False, seed=seed, obs_vel=not a.no_obs_vel,
                      tail='locked' if a.tail == 'locked' else 'active', ankle_clutch=a.ankle_clutch, couple_ankle=a.couple_ankle, render_mode='rgb_array' if render else None)
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
        for vx in a.turn_cmd if a.turns else []:
            for wz in a.turns:
                eps = []
                for ep in range(a.episodes):
                    env = new_env(3000+ep)
                    model, venv = load(a.model, env)
                    eps.append(episode(model, venv, env, [vx, 0., 0.], a.seconds, lock_tail=mode == 'locked',
                                       turn=(wz, a.turn_at, a.turn_s)))
                ok = [e for e in eps if not e['fell'] and e['turn_rad'] is not None]
                row = {'tail': mode, 'turn_cmd_rad_s': wz, 'turn_s': a.turn_s, 'cmd_vx': vx,
                       'falls': sum(e['fell'] for e in eps), 'episodes': len(eps),
                       'turn_rad_cmd': round(wz*a.turn_s, 2),
                       'turn_rad': round(float(np.mean([e['turn_rad'] for e in ok])), 2) if ok else None,
                       'turn_min_vx': round(float(np.mean([e['turn_min_vx'] for e in ok])), 2) if ok else None,
                       'turn_max_tilt': round(float(np.mean([e['turn_max_tilt'] for e in eps if e['turn_max_tilt'] is not None])), 2)}
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
