"""Evidence 195 chart: stage 1a gate of the 0.80 m leg policy l80_s1p against the R-03 pass r3_s1ho (largest drift per
condition, falls marked), and side frames of l80_s1p during a 0.8 m/s backward push.
Usage: .venv-sim/bin/python docs/evidence/195-l80-stand/plot.py -> media/gate_compare.png, media/push_back_frames.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT/'sim/rl'))
import stage_gates as sg  # noqa: E402


def rows(p):
    out = {}
    for l in Path(p).read_text().splitlines():
        d = json.loads(l)
        key = d['cond'] if d['cond'] != 'push' else f"{d['dir']} {d['dv']}"
        out[key] = (d['drift_max_m'], d['falls'], d['n'])
    return out


A = rows(OUT/'eval/gate_l80_s1p.jsonl')
B = rows(ROOT/'docs/evidence/188-stage1a-stand/eval/gate_r3_s1ho.jsonl')
keys = list(A)
fig, ax = plt.subplots(figsize=(13, 4.2))
x = np.arange(len(keys))
for k, (lab, R, c) in enumerate((('r3_s1ho, R-03 leg 0.92 m', B, '#9cc3d5'), ('l80_s1p, frozen leg 0.80 m', A, '#123f5c'))):
    v = [R[q][0] if q in R else np.nan for q in keys]
    ax.bar(x+(k-.5)*.38, v, .38, color=c, label=lab)
    for xi, q in zip(x, keys):
        if q in R and R[q][1]:
            ax.text(xi+(k-.5)*.38, R[q][0]+.02, f'{R[q][1]} fall', ha='center', fontsize=7, color='#b91c1c')
ax.axhline(.5, color='#b91c1c', ls='--', lw=1); ax.text(len(keys)-.5, .51, 'nominal drift limit 0.5 m', color='#b91c1c', fontsize=8, ha='right')
ax.set_xticks(x); ax.set_xticklabels(keys, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('largest horizontal drift (m)'); ax.legend(fontsize=8, loc='upper left')
ax.set_title('Stage 1a gate (30 s hold nominal 8 / randomized 24, pushes 8 per direction and size), MuJoCo + RobStride realism (not hardware)', fontsize=9)
fig.tight_layout(); fig.savefig(OUT/'media/gate_compare.png', dpi=130)

# frames of a 0.8 m/s backward push (gate seed 6200, kick at 5 s)
d = ROOT/'sim/rl/runs/l80_s1p'
a = json.loads((d/'args.json').read_text())
kw = {**sg.env_kwargs(a), 'level': 0., 'episode_s': 15., 'disturb_items': [], 'roll_impulse': None, 'yaw_impulse': None,
      'cmd_max': [0., 0., 0.], 'init_speed': False}
env = sg.RunEnv(**kw, randomize=False, seed=6200, render_mode='rgb_array'); env.init_seed = 6200
model, venv = sg.load(str(d/'model.zip'), env); venv.reset(); env.command = np.zeros(3); env.resample_steps = 10**9
env.model.vis.headlight.ambient[:] = .45
obs = venv.normalize_obs(env._obs()[None]); dd, b = env.data, env.base
r = mujoco.Renderer(env.model, 360, 420)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = b; cam.distance, cam.elevation, cam.azimuth = 2.3, -18., 90.
shots, times = [], (4.9, 5.2, 5.5, 6.0, 7.0, 10.0)
for k in range(500):
    if k == 250:
        R = dd.xmat[b].reshape(3, 3); dd.qvel[:2] += R[:2, :2] @ np.array([-.8, 0.])
    raw, *_ = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None])
    if any(abs((k+1)*.02-t) < 1e-6 for t in times):
        r.update_scene(dd, cam); shots.append(r.render()[75:330].copy())
fig, ax = plt.subplots(1, len(shots), figsize=(16, 3.4))
for i, (s, t) in enumerate(zip(shots, times)):
    ax[i].imshow(s); ax[i].axis('off'); ax[i].set_title(f't = {t:.1f} s' + (' (push at 5.0 s)' if i == 0 else ''), fontsize=9)
fig.suptitle('l80_s1p, backward push 0.8 m/s at 5 s (gate seed 6200), side view', fontsize=10)
fig.tight_layout(); fig.savefig(OUT/'media/push_back_frames.png', dpi=110)
