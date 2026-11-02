"""Evidence 190 figures: stand gate and stance of the three stand policies (media/nominal_refit.png) and a side render of
each policy's stance at t = 10 s, seed 6000 (media/stances.png).
Usage: .venv-sim/bin/python docs/evidence/190-r03-nominal-refit/plot.py"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'sim/rl'))
RUNS = ['r3_s1ho', 'r3_s1c', 'r3_s1n']
LAB = {'r3_s1ho': 'r3_s1ho\n(1a pass, old pose)', 'r3_s1c': 'r3_s1c\n(+6M, old pose)', 'r3_s1n': 'r3_s1n\n(+6M, refit pose)'}
COL = {'r3_s1ho': '#9aa5b1', 'r3_s1c': '#3b6e8f', 'r3_s1n': '#c2410c'}


def chart():
    gate = {r: {(g['cond'], g.get('dir'), g.get('dv')): g for g in map(json.loads, (OUT/f'eval/gate_{r}.jsonl').read_text().splitlines())}
            for r in RUNS}
    st = [json.loads(l) for l in (OUT/'eval/stance.jsonl').read_text().splitlines()]
    mean = lambda r, f: np.mean([f(s) for s in st if s['run'] == r])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(RUNS))
    for i, (cond, t) in enumerate([('hold', '30 s, nominal (8)'), ('hold_dr', '30 s, randomized (24)')]):
        ax[0].bar(x+(i-.5)*.38, [gate[r][(cond, None, None)]['drift_max_m'] for r in RUNS], .38,
                  color=[COL[r] for r in RUNS], alpha=1 if i == 0 else .55, label=t)
    ax[0].axhline(.5, color='#b91c1c', ls='--', lw=1); ax[0].text(2.5, .52, 'limit 0.5 m (nominal)', color='#b91c1c', ha='right', fontsize=8)
    ax[0].set_ylabel('largest base drift (m)'); ax[0].set_title('Stage 1a drift (solid nominal, light randomized)', fontsize=9)
    ax[1].bar(x, [mean(r, lambda s: s['power_w']) for r in RUNS], .55, color=[COL[r] for r in RUNS])
    ax[1].set_ylabel('mean |mechanical power| (W)'); ax[1].set_title('Standing still, 5-30 s, 4 seeds', fontsize=9)
    w = .27
    for j, k in enumerate(('knee', 'hip_pitch', 'hip_roll')):
        ax[2].bar(x+(j-1)*w, [mean(r, lambda s: s['tau_abs'][k]) for r in RUNS], w, label=k, color=['#334155', '#64748b', '#cbd5e1'][j])
    ax[2].set_ylabel('mean |torque| (N·m)'); ax[2].set_title('Joint torque standing still', fontsize=9); ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([LAB[r] for r in RUNS], fontsize=8)
    ax[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('R-03 stand policies: old nominal pose (hip -0.53) vs pose refit by the CoM-over-toes rule (hip -0.13), MuJoCo, not hardware', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/'media/nominal_refit.png', dpi=130)


def renders():
    import mujoco
    from evaluate import load
    from run_env import RunEnv
    from train_run import env_kwargs
    imgs = []
    for r in RUNS:
        a = json.loads((ROOT/f'sim/rl/runs/{r}/args.json').read_text())
        kw = {**env_kwargs(a), 'level': 0., 'episode_s': 15., 'disturb_items': [], 'roll_impulse': None, 'yaw_impulse': None,
              'cmd_max': [0., 0., 0.], 'init_speed': False}
        env = RunEnv(**kw, randomize=False, seed=6000, render_mode='rgb_array')
        env.init_seed = 6000
        model, venv = load(str(ROOT/f'sim/rl/runs/{r}/model.zip'), env)
        venv.reset(); env.command = np.zeros(3); env.resample_steps = 10**9
        obs = venv.normalize_obs(env._obs()[None])
        for _ in range(500):
            raw, *_ = env.step(model.predict(obs, deterministic=True)[0][0]); obs = venv.normalize_obs(raw[None])
        env.model.vis.headlight.ambient[:] = .45
        rd = mujoco.Renderer(env.model, 420, 420)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = env.base
        R = env.data.xmat[env.base].reshape(3, 3)
        cam.distance, cam.elevation = 2.2, -14.
        cam.azimuth = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))+90.
        rd.update_scene(env.data, cam); imgs.append(rd.render()[130:].copy())  # drop the sky band
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, im, r in zip(ax, imgs, RUNS):
        a.imshow(im); a.axis('off'); a.set_title(LAB[r].replace('\n', ' '), fontsize=9)
    fig.suptitle('Stance at t = 10 s, zero command, seed 6000 (side view, nose to the right)', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/'media/stances.png', dpi=120)


if __name__ == '__main__':
    chart(); renders()
