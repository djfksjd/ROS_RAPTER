"""Evidence 190 probe: the stance a stand policy actually holds (30 s zero command, randomize off, gate seeds 6000-6003):
joint angles, whole-body CoM x minus the centre of the loaded toe contacts (heading frame), base height, mean |torque| per
joint type and the summed mechanical |power|, averaged over 5-30 s.
Usage: .venv-sim/bin/python docs/evidence/190-r03-nominal-refit/stance_probe.py RUN [RUN ...] -> eval/stance.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'sim/rl'))
from evaluate import load  # noqa: E402
from run_env import RunEnv  # noqa: E402
from train_run import env_kwargs  # noqa: E402

RUNS = ROOT/'sim/rl/runs'


def stance(run, seed, seconds=30.):
    a = json.loads((RUNS/run/'args.json').read_text())
    kw = {**env_kwargs(a), 'level': 0., 'episode_s': seconds+5., 'disturb_items': [], 'roll_impulse': None,
          'yaw_impulse': None, 'cmd_max': [0., 0., 0.], 'init_speed': False}
    env = RunEnv(**kw, randomize=False, seed=seed)
    env.init_seed = seed
    model, venv = load(str(RUNS/run/'model.zip'), env)
    venv.reset()
    env.command = np.zeros(3); env.resample_steps = 10**9
    obs = venv.normalize_obs(env._obs()[None])
    m, d, b = env.model, env.data, env.base
    names = [m.joint(m.actuator_trnid[i, 0]).name for i in env.act]
    rows, fell = [], None
    for k in range(int(seconds/.02)):
        raw, _, term, _, _ = env.step(model.predict(obs, deterministic=True)[0][0])
        obs = venv.normalize_obs(raw[None])
        if term:
            fell = round((k+1)*.02, 2)
            break
        if k*.02 < 5.:
            continue
        R = d.xmat[b].reshape(3, 3)
        fwd = R[:2, 0]/np.linalg.norm(R[:2, 0])
        pts = [c.pos[:2] for c in d.contact[:d.ncon] if 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])]
        com = d.subtree_com[env.root][:2]
        tau, qd = d.actuator_force[env.act], d.qvel[env.v_adr]
        rows.append(dict(com_minus_support=float(fwd @ (com-np.mean(pts, 0))) if pts else np.nan,
                         q=d.qpos[env.q_adr].copy(), tau=np.abs(tau), power=float(np.abs(tau*qd).sum()),
                         height=float(d.xpos[b][2]), pitch=float(np.arcsin(-R[2, 0]))))
    q = np.mean([r['q'] for r in rows], 0)
    tau = np.mean([r['tau'] for r in rows], 0)
    by = lambda arr, key: float(np.mean([v for n, v in zip(names, arr) if key in n]))
    return dict(run=run, seed=seed, fell=fell, nominal=a.get('nominal'),
                hip_pitch=round(by(q, 'hip_pitch'), 3), knee=round(by(q, 'knee'), 3),
                com_minus_support_m=round(float(np.nanmean([r['com_minus_support'] for r in rows])), 4),
                height_m=round(float(np.mean([r['height'] for r in rows])), 3),
                torso_pitch_rad=round(float(np.mean([r['pitch'] for r in rows])), 3),
                tau_abs={k: round(by(tau, k), 2) for k in ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'tail')},
                power_w=round(float(np.mean([r['power'] for r in rows])), 1))


if __name__ == '__main__':
    with open(OUT/'eval/stance.jsonl', 'a') as f:
        for run in sys.argv[1:]:
            for seed in range(6000, 6004):
                r = stance(run, seed)
                print(json.dumps(r)); f.write(json.dumps(r)+'\n')
