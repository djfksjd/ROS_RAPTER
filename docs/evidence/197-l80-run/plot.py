"""증거 197: 저장된 게이트 그래프와 6 m/s 정책 재생 장면(학습·장시간 게이트 없음).
실행: .venv-sim/bin/python docs/evidence/197-l80-run/plot.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
MEDIA = OUT / 'media'
POLICIES = [('l80_r1_4M', 'l80_r1 4M', '#777777', 's'),
            ('l80_r1_6M', 'l80_r1 6M (adopted)', '#174e70', 'o'),
            ('l80_r1', 'l80_r1 8M', '#aa7530', '^'),
            ('l80_r3', 'l80_r3', '#866183', 'D')]


def candidates():
    fig, (speed, heat) = plt.subplots(1, 2, figsize=(13, 5.5))
    derate = heat.twinx()
    x = np.array([2., 6.])
    speed.fill_between(x, x*.9, x*1.1, color='#dddddd', label='2/4/6: +/-10% band')
    speed.axhline(6.5, color='#333333', ls=':', lw=1, label='8 command: 6.5 minimum')
    missing = []
    for name, label, color, marker in POLICIES:
        rows = [json.loads(line) for line in (OUT / f'eval/gate_run_{name}.jsonl').read_text().splitlines()]
        sr = [r for r in rows if r['cond'] == 'speed']
        means = []
        for r in sr:
            v = [e['v_mean'] for e in r['episodes'] if e['v_mean'] is not None]
            means.append(np.mean(v) if v else np.nan)
            speed.scatter([r['cmd']]*len(v), v, color=color, s=13, alpha=.45)
            if r['falls']:
                missing.append(f"{label.split(' (')[0]}: {r['cmd']:g} ({r['falls']}/{r['n']})")
        speed.plot([r['cmd'] for r in sr], means, color=color, marker=marker, ms=5, lw=1.2, label=label)
        hr = [r for r in rows if r['cond'] == 'heat' and r['mount'] == 'plate']
        heat.plot([r['cmd'] for r in hr], [r['t_winding_max_c'] for r in hr], color=color, marker=marker, lw=1.2)
        derate.plot([r['cmd'] for r in hr], [r['min_derate'] for r in hr], color=color, marker=marker, ls='--', lw=1.2, mfc='white')
    speed.set(xlabel='Forward command (m/s)', ylabel='Mean speed, 10-20 s (m/s)',
              title='Speed: 4 episodes per command', xticks=[2, 4, 6, 8, 10, 11.1], ylim=(0, 9))
    speed.legend(fontsize=8, loc='lower right', frameon=False)
    heat.set(xlabel='Forward command (m/s)', ylabel='Max winding temperature (C), solid',
             title='10 min heat: plate mount', xticks=[4, 5, 6], xlim=(3.85, 6.15), ylim=(70, 145))
    derate.set(ylabel='Minimum derate factor, dashed (1 = none)', ylim=(.4, 1.08))
    heat.text(.04, .96, '8M: only command 6 recorded; 4/5 unmeasured', transform=heat.transAxes, fontsize=8, va='top')
    for ax in (speed, heat):
        ax.grid(axis='y', color='#eeeeee', lw=.7)
        ax.spines['top'].set_visible(False)
    derate.spines['top'].set_visible(False)
    fig.suptitle('Stage 3 candidates | MuJoCo simulation | 6M adopted; stage 3 FAIL, user decision pending', fontsize=11)
    fig.text(.02, .07, 'Dots: completed episodes; lines: their means. Fallen episodes have no speed value; all-fall commands leave gaps.', fontsize=8)
    fig.text(.02, .035, 'Falls (command, falls/4): ' + '; '.join(missing), fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, .94))
    fig.savefig(MEDIA / 'run_candidates.png', dpi=150)
    plt.close(fig)


def frames():
    import mujoco
    sys.path.insert(0, str(ROOT / 'sim/rl'))
    import stage_gates as sg

    d = ROOT / 'sim/rl/runs/l80_r1_6M'
    a = json.loads((d / 'args.json').read_text())
    kw = {**sg.env_kwargs(a), 'level': 0., 'episode_s': 15., 'disturb_items': [],
          'roll_impulse': None, 'yaw_impulse': None, 'cmd_max': [0., 0., 0.], 'init_speed': False}
    env = sg.RunEnv(**kw, randomize=False, seed=6700, render_mode='rgb_array')
    env.init_seed = 6700
    model, venv = sg.load(str(d / 'model.zip'), env)
    venv.reset()
    env.resample_steps = 10**9
    env.command = np.array([6., 0., 0.])
    env.model.vis.headlight.ambient[:] = .45
    obs = venv.normalize_obs(env._obs()[None])
    try:
        renderer = mujoco.Renderer(env.model, 360, 420)
    except mujoco.cgl.cgl.CGLError:
        renderer = None
        print('OpenGL unavailable; rendering actual MuJoCo scene with NumPy/Matplotlib CPU.', flush=True)
    scene = mujoco.MjvScene(env.model, maxgeom=1000)
    snapshots = []
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.base
    cam.distance, cam.elevation, cam.azimuth = 2.3, -18., 90.
    steps = [300 + 5*i for i in range(8)]
    shots = []
    try:
        for step in range(1, steps[-1]+1):
            raw, _, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0][0])
            if term or trunc:
                raise RuntimeError(f'Replay ended at {step*.02:.2f} s')
            obs = venv.normalize_obs(raw[None])
            if step in steps:
                if renderer is not None:
                    renderer.update_scene(env.data, cam)
                    shots.append(renderer.render()[75:330].copy())
                else:
                    mujoco.mjv_updateScene(env.model, env.data, mujoco.MjvOption(), None, cam,
                                           mujoco.mjtCatBit.mjCAT_ALL, scene)
                    snapshots.append(dict(
                        geoms=[dict(type=int(g.type), size=g.size.tolist(), pos=g.pos.tolist(),
                                    mat=g.mat.reshape(3, 3).tolist(), rgba=g.rgba.tolist())
                               for g in scene.geoms[:scene.ngeom] if g.rgba[3] > 0],
                        camera_pos=np.mean([c.pos for c in scene.camera], axis=0).tolist(),
                        camera_forward=scene.camera[0].forward.tolist(),
                        camera_up=scene.camera[0].up.tolist(),
                        fovy=float(env.model.vis.global_.fovy)))
    finally:
        if renderer is not None:
            renderer.close()
        venv.close()
    if snapshots:
        from render_cpu import render
        # 증거 196과 같은 행을 자른다. 실제 지오메트리·자세·카메라를 사용한다.
        shots = [render(snapshot)[75:330] for snapshot in snapshots]
    assert len(shots) == 8
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
    for ax, shot, step in zip(axes.flat, shots, steps):
        ax.imshow(shot)
        ax.axis('off')
        ax.set_title(f't = {step*.02:.1f} s', fontsize=10)
    backend = 'CPU projection of MuJoCo scene' if snapshots else 'MuJoCo OpenGL'
    fig.suptitle('l80_r1_6M | command 6 m/s | seed 6700 | side view, 0.1 s apart\n' + backend + '; actual policy replay, illustration only', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .91), h_pad=2.0)
    fig.savefig(MEDIA / 'run_frames.png', dpi=150)
    plt.close(fig)
    print('Saved 8 simulation frames (6.0-6.7 s), no early termination.')


if __name__ == '__main__':
    MEDIA.mkdir(exist_ok=True)
    candidates()
    frames()
