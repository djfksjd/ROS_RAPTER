"""Evidence 189: R-03 model check figure, per-start get-up success of the fall curriculum runs, get-up renders.

  model    side-view geometry of R-03 at the training stand pose (segment lengths from the MuJoCo body/joint positions),
           masses per part, joint ranges -> media/r03_model_check.png
  rates    each fall-curriculum run (r3_gfa..d) on 20 fixed fall starts (seeds 7000+, randomize off, strict 1 s hold,
           no training leak) at its own fall_t and at 2.5 s: start tilt and success per start -> eval/fall_rates.json
  render   the 1b gate rollout of r3_gfd (stage_gates.getup_rollout: get-up policy, then the 1a stand policy r3_s1ho
           for 10 s): sit seed 7005 (the one success) and side seed 7000 (a failure) -> media/gfd_gate_*.gif/mp4
Usage: .venv-sim/bin/python docs/evidence/189-stage1b-getup-feasibility/getup_probe.py model|rates|render
"""
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNS = ROOT/'sim/rl/runs'
sys.path.insert(0, str(ROOT/'sim/rl'))
from evaluate import load  # noqa: E402
from run_env import RunEnv  # noqa: E402
from train_run import env_kwargs  # noqa: E402

STAGES = (('r3_gfa', .6), ('r3_gfb', 1.), ('r3_gfc', 1.6), ('r3_gfd', 2.5))
EVAL_KW = {'level': 0., 'episode_s': 15., 'disturb_items': [], 'roll_impulse': None, 'yaw_impulse': None,
           'cmd_max': [0., 0., 0.], 'init_speed': False}


def tilt_of(env):
    return float(np.arccos(np.clip(env.data.xmat[env.base].reshape(3, 3)[2, 2], -1, 1)))


def episode(run, seed, fall_t, fall_v=None, frames=None, seconds=10.):
    a = json.loads((RUNS/run/'args.json').read_text())
    g = {**json.loads(a['getup']), 'modes': {'fall': 1.}, 'fall_t': fall_t, 'leak': None}
    if fall_v is not None:
        g['fall_v'] = fall_v
    env = RunEnv(**{**env_kwargs(a), **EVAL_KW, 'getup': g}, randomize=False, seed=seed)
    env.init_seed = seed
    model, venv = load(str(RUNS/run/'model.zip'), env)
    venv.reset()
    start = tilt_of(env)
    obs = venv.normalize_obs(env._obs()[None])
    if frames is not None:
        r = mujoco.Renderer(env.model, 360, 480)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = env.base
        cam.distance, cam.elevation, cam.azimuth = 2.4, -12., 135.
    t_up, trace = None, []
    for k in range(int(seconds/.02)):
        raw, _, _, _, info = env.step(model.predict(obs, deterministic=True)[0][0])
        obs = venv.normalize_obs(raw[None])
        trace.append((round(tilt_of(env), 3), round(float(env.data.xpos[env.base][2]), 3)))
        if frames is not None and k % 2 == 0:
            r.update_scene(env.data, cam); frames.append(r.render().copy())
        if info['getup']['success'] and t_up is None:
            t_up = round((k+1)*.02, 2)
            if frames is None:
                break
    return dict(seed=seed, start_tilt=round(start, 3), t_up=t_up, min_tilt=min(t for t, _ in trace)), trace


def rates():
    res = {}
    for run, ft in STAGES:
        if not (RUNS/run/'model.zip').exists():
            continue
        for f in sorted({ft, 2.5}):
            rows = [episode(run, 7000+i, f)[0] for i in range(20)]
            res[f'{run}@{f}'] = rows
            print(run, f, sum(r['t_up'] is not None for r in rows), '/ 20', flush=True)
    (OUT/'eval/fall_rates.json').write_text(json.dumps(res))


def model():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = json.loads((RUNS/'r3_s1ho/args.json').read_text())
    env = RunEnv(**env_kwargs(a), randomize=False, seed=1)
    env.reset(seed=1)
    m, d = env.model, env.data
    pos = lambda n: d.xpos[m.body(n).id]
    anc = lambda n: d.xanchor[m.joint(n).id]
    pts = {'hip': anc('left_hip_pitch_joint'), 'knee': anc('left_knee_pitch_joint'),
           'ankle': anc('left_ankle_pitch_joint'), 'foot': anc('left_ankle_roll_joint')}
    seg = {'thigh (hip-knee)': ('hip', 'knee'), 'shin (knee-ankle)': ('knee', 'ankle'),
           'metatarsus (ankle-foot)': ('ankle', 'foot')}
    L = {k: float(np.linalg.norm(pts[b]-pts[a_])) for k, (a_, b) in seg.items()}
    tail_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in (m.body('tail_yaw_link').id, m.body('tail_link').id)]
    tail_x = min(d.geom_xpos[g][0]-np.abs(d.geom_xmat[g].reshape(3, 3)[0]) @ m.geom_size[g] for g in tail_geoms)
    tail_len = float(anc('tail_yaw_joint')[0]-tail_x)
    body_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == env.base]
    ext = [(d.geom_xpos[g][0]-np.abs(d.geom_xmat[g].reshape(3, 3)[0]) @ m.geom_size[g],
            d.geom_xpos[g][0]+np.abs(d.geom_xmat[g].reshape(3, 3)[0]) @ m.geom_size[g]) for g in body_geoms]
    body_x = (float(min(e[0] for e in ext)), float(max(e[1] for e in ext)))
    body_len = body_x[1]-body_x[0]
    parts = {'torso (base)': m.body_mass[env.base],
             'hip roll links (2, incl. motors)': 2*m.body_mass[m.body('left_hip_roll_link').id],
             'thighs (2)': 2*m.body_mass[m.body('left_thigh_link').id],
             'shins + feet + toes (2)': 2*m.body_subtreemass[m.body('left_shin_link').id],
             'tail (yaw link + tail)': m.body_subtreemass[m.body('tail_yaw_link').id]}
    total = float(m.body_subtreemass[0])

    fig = plt.figure(figsize=(15, 5.8))
    ax = fig.add_axes([.01, .1, .38, .8])
    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [-.2, 0., .45]; cam.distance, cam.elevation, cam.azimuth = 2.4, -8., -90.
    m.vis.headlight.ambient[:] = .45; m.vis.headlight.diffuse[:] = .6
    r.update_scene(d, cam)
    ax.imshow(r.render()); ax.axis('off')
    ax.set_title(f'R-03 at the stand pose (MuJoCo render, not hardware)\nbase height {d.xpos[env.base][2]:.2f} m, '
                 f'hip height {pts["hip"][2]:.2f} m', fontsize=10)

    ax2 = fig.add_axes([.43, .1, .22, .8])
    for (k, (a_, b)), c in zip(seg.items(), ('#1f6f8b', '#c06014', '#5b8c3a')):
        P = np.array([pts[a_], pts[b]])
        ax2.plot(P[:, 0], P[:, 2], '-o', color=c, lw=4, label=f'{k}: {L[k]*100:.1f} cm')
    ax2.plot(body_x, [pts['hip'][2]+.08]*2, color='#444', lw=6,
             label=f'torso length {body_len*100:.0f} cm')
    ax2.plot([anc('tail_yaw_joint')[0], tail_x], [anc('tail_yaw_joint')[2]]*2, color='#888', lw=3,
             label=f'tail {tail_len*100:.0f} cm (2 axes, ±0.8 rad)')
    ax2.axhline(0, color='k', lw=.8)
    ax2.set_aspect('equal'); ax2.set_xlabel('x (m), front = + (render: front = left)'); ax2.set_ylabel('z (m)')
    ax2.legend(fontsize=7.5, loc='lower left', bbox_to_anchor=(0., -.02)); ax2.set_ylim(-.05, 1.)
    ax2.set_title(f'left leg, side view; leg sum {sum(L.values())*100:.0f} cm', fontsize=10)

    ax3 = fig.add_axes([.82, .56, .16, .34])
    names, vals = list(parts), [float(v) for v in parts.values()]
    ax3.barh(names, vals, color='#1f6f8b')
    for i, v in enumerate(vals):
        ax3.text(v+.1, i, f'{v:.2f}', va='center', fontsize=8)
    ax3.invert_yaxis(); ax3.set_xlabel('kg'); ax3.tick_params(labelsize=8)
    ax3.set_title(f'mass {total:.1f} kg', fontsize=10)
    ax4 = fig.add_axes([.7, .04, .29, .38]); ax4.axis('off')
    rng = lambda n: m.jnt_range[m.joint(n).id]
    rows = [['hip yaw', 'RS06 1.5:1', '±0.5'], ['hip roll', 'RS06 2.5:1', '±0.5'],
            ['hip pitch', 'RS04 0.85:1', f'{rng("left_hip_pitch_joint")[0]:.1f}..{rng("left_hip_pitch_joint")[1]:.1f}'],
            ['knee', 'RS04 4-bar', f'{env.knee_band[0]:.2f}..{env.knee_band[1]:.2f}'],
            ['ankle pitch', 'coupled to knee', '-'], ['ankle roll', 'locked', '-'],
            ['tail yaw / pitch', 'RS06 / RS03', '±0.8']]
    t = ax4.table(cellText=rows, colLabels=['axis', 'motor', 'range (rad)'], loc='center', cellLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.25)
    ax4.set_title('10 policy axes (4 per leg + 2 tail)', fontsize=10)
    fig.savefig(OUT/'media/r03_model_check.png', dpi=130)
    print(json.dumps(dict(segments_m={k: round(v, 3) for k, v in L.items()}, torso_m=round(body_len, 3),
                          tail_m=round(tail_len, 3), mass_kg=round(total, 2),
                          parts_kg={k: round(float(v), 2) for k, v in parts.items()},
                          base_h=round(float(d.xpos[env.base][2]), 3), hip_h=round(float(pts['hip'][2]), 3))))


def render():
    import stage_gates
    for name, mode, seed in (('gfd_gate_sit7005', 'sit', 7005), ('gfd_gate_side7000', 'side', 7000)):
        frames, step = [], RunEnv.step
        state = {}

        def rec(self, action):
            out = step(self, action)
            if 'r' not in state:
                self.model.vis.headlight.ambient[:] = .5; self.model.vis.headlight.diffuse[:] = .7  # visual only
                state['r'] = mujoco.Renderer(self.model, 360, 480)
                cam = state['cam'] = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = self.base; cam.distance, cam.elevation, cam.azimuth = 2.6, -12., 135.
            if self.steps % 2 == 0:
                state['r'].update_scene(self.data, state['cam']); frames.append(state['r'].render().copy())
            return out
        RunEnv.step = rec
        try:
            row = stage_gates.getup_rollout(RUNS/'r3_gfd', RUNS/'r3_s1ho', mode, seed)
        finally:
            RunEnv.step = step
        print(name, row, len(frames), flush=True)
        raw = OUT/f'media/{name}.rgb'
        np.concatenate(frames).tofile(raw)
        h, w = frames[0].shape[:2]
        base = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '25', '-i', str(raw)]
        subprocess.run(base+['-pix_fmt', 'yuv420p', str(OUT/f'media/{name}.mp4')], check=True)
        subprocess.run(base+['-vf', 'fps=12.5,scale=360:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                             str(OUT/f'media/{name}.gif')], check=True)
        raw.unlink()


if __name__ == '__main__':
    {'model': model, 'rates': rates, 'render': render}[sys.argv[1]]()
