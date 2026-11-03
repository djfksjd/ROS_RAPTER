"""Evidence 196 media: stage 2 walk gate of l80_w2 (commanded vs measured speed and yaw rate per episode) and side
frames of l80_w2 walking at 1.0 m/s.
Usage: .venv-sim/bin/python docs/evidence/196-l80-walk/plot.py -> media/walk_gate.png, media/walk_frames.png"""
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

rows = [json.loads(l) for l in (OUT/'eval/gate_walk_l80_w2.jsonl').read_text().splitlines()]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
tr = [r for r in rows if r['cond'] == 'track']
for r in tr:
    v = [e['v_mean'] for e in r['episodes']]
    ax[0].scatter([r['cmd']]*len(v), v, color='#123f5c', s=18, zorder=3)
    tol = max(.1, .15*r['cmd'])
    ax[0].plot([r['cmd']-.04, r['cmd']+.04], [r['cmd']-tol]*2, color='#b91c1c', lw=1)
    ax[0].plot([r['cmd']-.04, r['cmd']+.04], [r['cmd']+tol]*2, color='#b91c1c', lw=1)
ax[0].plot([0, 1.6], [0, 1.6], color='#999', ls=':', lw=1)
ax[0].set_xlabel('forward command (m/s)'); ax[0].set_ylabel('mean forward speed 5-15 s (m/s)')
ax[0].set_title('Speed tracking, 4 seeds each (red: pass band)', fontsize=9)
tu = [r for r in rows if r['cond'] == 'turn']
for r in tu:
    w = [e['yaw_rate_mean'] for e in r['episodes']]
    ax[1].scatter([r['cmd']]*len(w), w, color='#123f5c', s=18, zorder=3)
    ax[1].plot([r['cmd']-.05, r['cmd']+.05], [.75*r['cmd']]*2, color='#b91c1c', lw=1)
    ax[1].plot([r['cmd']-.05, r['cmd']+.05], [1.25*r['cmd']]*2, color='#b91c1c', lw=1)
ax[1].axhline(0, color='#ccc', lw=.8)
ax[1].set_xlabel('yaw command (rad/s), forward 0'); ax[1].set_ylabel('mean yaw rate 3-10 s (rad/s)')
ax[1].set_title('Turn in place (red: pass band)', fontsize=9)
ss = next(r for r in rows if r['cond'] == 'stopstart')
st = [v for e in ss['episodes'] for v in e['speed_2s_after_stop']]
rc = [t for e in ss['episodes'] for t in e['t_reach_0p8']]
ax[2].bar(['reach 0.8 m/s\nafter start (s)', '|speed| 2 s\nafter stop (m/s)'], [max(rc), max(abs(v) for v in st)], color=['#5b9bbd', '#123f5c'])
ax[2].axhline(.15, color='#b91c1c', ls='--', lw=1); ax[2].text(1.4, .16, 'stop limit 0.15', color='#b91c1c', fontsize=8, ha='right')
ax[2].set_title('Stop / start, worst of 4 seeds (start limit 2 s)', fontsize=9)
fig.suptitle('Stage 2 walk gate, l80_w2 (0.80 m leg), MuJoCo + RobStride realism (not hardware); all nominal 0 falls, randomized 1.0 m/s 0/24', fontsize=10)
fig.tight_layout(); fig.savefig(OUT/'media/walk_gate.png', dpi=130)

d = ROOT/'sim/rl/runs/l80_w2'
a = json.loads((d/'args.json').read_text())
kw = {**sg.env_kwargs(a), 'level': 0., 'episode_s': 15., 'disturb_items': [], 'roll_impulse': None, 'yaw_impulse': None,
      'cmd_max': [0., 0., 0.], 'init_speed': False}
env = sg.RunEnv(**kw, randomize=False, seed=6300, render_mode='rgb_array'); env.init_seed = 6300
model, venv = sg.load(str(d/'model.zip'), env); venv.reset(); env.resample_steps = 10**9; env.command = np.array([1., 0., 0.])
env.model.vis.headlight.ambient[:] = .45
obs = venv.normalize_obs(env._obs()[None]); dd, b = env.data, env.base
r = mujoco.Renderer(env.model, 360, 420)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = b; cam.distance, cam.elevation, cam.azimuth = 2.3, -18., 90.
times = [6.0+.1*i for i in range(8)]
shots = []
for k in range(400):
    raw, *_ = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None])
    if any(abs((k+1)*.02-t) < 1e-6 for t in times):
        r.update_scene(dd, cam); shots.append(r.render()[75:330].copy())
fig, ax = plt.subplots(1, len(shots), figsize=(18, 3.2))
for i, (s, t) in enumerate(zip(shots, times)):
    ax[i].imshow(s); ax[i].axis('off'); ax[i].set_title(f't = {t:.1f} s', fontsize=9)
fig.suptitle('l80_w2 walking at a 1.0 m/s command (seed 6300), side view tracking the body, 0.1 s apart', fontsize=10)
fig.tight_layout(); fig.savefig(OUT/'media/walk_frames.png', dpi=110)
