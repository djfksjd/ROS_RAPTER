"""Refit the parallel springs to a trained policy's own joint-torque trajectories (arch8, evidence 86 §17).

Records deterministic rollouts at several commands, reconstructs the torque each drive must deliver without springs
(tau_required = motor torque + current passive spring torque on that drive; the 1:1 knee-ankle coupling maps the
ankle spring onto the knee drive with a minus sign), then searches a new parallel spring per drive - stiffness k, rest
angle q0, mode (bilateral / unilateral +/-), engaged always, only in stance (clutch) or only in swing - that minimises
the drive's motor RMS torque (the heat criterion) without raising its peak. Output: JSON with before/after RMS and the
spring set to pass to RunEnv(springs_override=...).
Usage: .venv-sim/bin/python sim/rl/spring_refit.py sim/rl/runs/run_arch8b/model.zip --out spring_refit.json
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from evaluate import load  # noqa: E402
from run_env import RunEnv  # noqa: E402


def record(model_path, cmds=(3., 4.5, 6., 9.3), seconds=8.):
    a = json.load(open(Path(model_path).with_name('args.json')))
    data = {s: {'knee': [], 'hip': []} for s in ('left', 'right')}
    for cmd in cmds:
        env = RunEnv(randomize=False, seed=1000, mass=a.get('mass', 11.36), arch8=True, ankle_clutch=True, kin=json.loads(a['kin']),
                     weights=json.loads(a['weights']), track_sigma_frac=a.get('track_sigma_frac', .3),
                     springs_override=json.loads(a.get('springs_override') or 'null'))
        model, venv = load(model_path, env)
        obs = venv.reset(); env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0
        obs = venv.normalize_obs(env._obs()[None])
        d = env.data
        for i in range(int(seconds/.02)):
            act, _ = model.predict(obs, deterministic=True)
            raw, _, term, _, info = env.step(act[0]); obs = venv.normalize_obs(raw[None])
            if i < 50:
                continue
            q, qd, tau = d.qpos[env.q_adr], d.qvel[env.v_adr], d.actuator_force[env.act]
            for s in ('left', 'right'):
                k, an, h = env.ix[f'{s}_knee_pitch_joint'], env.ix[f'{s}_ankle_pitch_joint'], env.ix[f'{s}_hip_pitch_joint']
                pk, pa, ph = d.qfrc_passive[env.v_adr[k]], d.qfrc_passive[env.v_adr[an]], d.qfrc_passive[env.v_adr[h]]
                st = bool(info['loaded'][s])
                data[s]['knee'].append((tau[k] + (pk - pa), q[k], qd[k], st))   # required knee-drive torque without springs
                data[s]['hip'].append((tau[h] + ph, q[h], qd[h], st))
            if term:
                break
    return {j: np.array(data['left'][j] + data['right'][j]) for j in ('knee', 'hip')}


def fit_touchdown_clutch(arr, k_max=400.):
    """Clutch that locks the spring at the touchdown angle (zero torque at engagement, BirdBot / ostrich ligament):
    spring torque = -k (q - q_td) during stance only, q_td = the angle at the start of each stance."""
    req, q, st = arr[:, 0], arr[:, 1], arr[:, 3].astype(bool)
    q_td = np.zeros_like(q); cur = q[0]
    for i in range(len(q)):
        if st[i] and (i == 0 or not st[i-1]):
            cur = q[i]
        q_td[i] = cur
    base_rms, base_pk = float(np.sqrt(np.mean(req**2))), float(np.abs(req).max())
    best = dict(rms=base_rms, peak=base_pk, k=0., mode='none')
    for mode in ('bi', 'uni+', 'uni-'):
        for k in np.linspace(5, k_max, 80):
            dq = q - q_td
            dq = np.maximum(dq, 0) if mode == 'uni+' else np.minimum(dq, 0) if mode == 'uni-' else dq
            motor = req + k*dq*st
            if np.abs(motor).max() > base_pk + 1e-9:
                continue
            r = float(np.sqrt(np.mean(motor**2)))
            if r < best['rms']:
                best = dict(rms=r, peak=float(np.abs(motor).max()), k=float(k), mode=mode)
    return dict(before=dict(rms=round(base_rms, 1), peak=round(base_pk, 1)), after={k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()})


def fit(arr, k_max=300.):
    req, q, qd, st = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3].astype(bool)
    base_rms, base_pk = float(np.sqrt(np.mean(req**2))), float(np.abs(req).max())
    best = dict(rms=base_rms, peak=base_pk, k=0., q0=0., mode='none', engaged='always')
    for engaged, mask in (('always', np.ones_like(st)), ('stance', st), ('swing', ~st)):
        for mode in ('bi', 'uni+', 'uni-'):
            for k in np.linspace(5, k_max, 60):
                for q0 in np.linspace(q.min() - .3, q.max() + .3, 50):
                    dq = q - q0
                    dq = np.maximum(dq, 0) if mode == 'uni+' else np.minimum(dq, 0) if mode == 'uni-' else dq
                    motor = req + k*dq*mask        # spring torque = -k dq (restoring); motor supplies the rest
                    if np.abs(motor).max() > base_pk + 1e-9:
                        continue
                    r = float(np.sqrt(np.mean(motor**2)))
                    if r < best['rms']:
                        best = dict(rms=r, peak=float(np.abs(motor).max()), k=float(k), q0=float(q0), mode=mode, engaged=engaged)
    return dict(before=dict(rms=round(base_rms, 1), peak=round(base_pk, 1)),
                after={k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model'); p.add_argument('--out')
    a = p.parse_args()
    arr = record(a.model)
    res = {j: dict(fixed_rest=fit(v), touchdown_clutch=fit_touchdown_clutch(v)) for j, v in arr.items()}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1)+'\n')


if __name__ == '__main__':
    main()
