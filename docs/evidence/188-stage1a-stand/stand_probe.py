"""Evidence 188 probes: base paths of the 30 s stand (gate seeds), odometry-error sensitivity of r3_s1ho, and a render.
Usage: .venv-sim/bin/python docs/evidence/188-stage1a-stand/stand_probe.py {paths|odom|render}
"""
import json
import subprocess
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


def episode(run, seed, seconds=30., randomize=False, kick=None, override=None, blind=False, frames=None):
    """stage_gates.rollout with env overrides; returns the base xy path (50 Hz). blind: the hold inputs read 0."""
    a = json.loads((RUNS/run/'args.json').read_text())
    kw = {**env_kwargs(a), 'level': 0., 'episode_s': seconds+5., 'disturb_items': [], 'roll_impulse': None,
          'yaw_impulse': None, 'cmd_max': [0., 0., 0.], 'init_speed': False, **(override or {})}
    env = RunEnv(**kw, randomize=randomize, seed=seed, render_mode='rgb_array' if frames is not None else None)
    env.init_seed = seed
    model, venv = load(str(RUNS/run/'model.zip'), env)
    venv.reset()
    env.command = np.zeros(3); env.resample_steps = 10**9
    if blind:
        env._hold_obs = lambda: np.zeros(3)
    obs = venv.normalize_obs(env._obs()[None])
    d, b = env.data, env.base
    x0 = d.xpos[b][:2].copy()
    path, fell = [], None
    if frames is not None:
        import mujoco
        r = mujoco.Renderer(env.model, 360, 480)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = b
        cam.distance, cam.elevation, cam.azimuth = 2.2, -12., 135.
    for k in range(int(seconds/.02)):
        if kick is not None and k == int(5./.02):
            R = d.xmat[b].reshape(3, 3)
            d.qvel[:2] += R[:2, :2] @ np.asarray(kick, float)
        raw, _, term, _, _ = env.step(model.predict(obs, deterministic=True)[0][0])
        obs = venv.normalize_obs(raw[None])
        path.append(d.xpos[b][:2]-x0)
        if frames is not None and k % 2 == 0:
            r.update_scene(d, cam); frames.append(r.render().copy())
        if term:
            fell = round((k+1)*.02, 2)
            break
    return np.array(path), fell


def paths():
    res = {}
    for run in ('r3_s1lp2', 'r3_s1ho'):
        res[run] = [episode(run, 6000+i)[0].tolist() for i in range(8)]
    (OUT/'eval/paths.json').write_text(json.dumps(res))


def odom():
    rows = []
    for name, ov, blind in (('noise 0.02, walk 0.005 (trained)', None, False),
                            ('walk 0.02', {'obs_hold': {'noise': .02, 'walk': .02}}, False),
                            ('walk 0.05', {'obs_hold': {'noise': .02, 'walk': .05}}, False),
                            ('noise 0.05, walk 0.02', {'obs_hold': {'noise': .05, 'walk': .02}}, False),
                            ('hold inputs 0 (blind)', None, True)):
        eps = [episode('r3_s1ho', 6000+i, override=ov, blind=blind) for i in range(8)]
        drift = [round(float(np.max(np.linalg.norm(p, axis=1))), 3) for p, _ in eps]
        rows.append(dict(cond=name, falls=sum(f is not None for _, f in eps), drift_max_m=max(drift), drift_m=drift))
        print(json.dumps(rows[-1]), flush=True)
    (OUT/'eval/odom_sensitivity.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')


def render():
    for name, kick in (('ho_hold30s', None), ('ho_push_back08', (-.8, 0.))):
        frames = []
        episode('r3_s1ho', 6000, seconds=30. if kick is None else 12., kick=kick, frames=frames)
        if kick is None:
            frames = frames[::3]  # 30 s at 3x speed
        raw = OUT/f'media/{name}.rgb'
        np.concatenate(frames).tofile(raw)
        h, w = frames[0].shape[:2]
        base = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '25', '-i', str(raw)]
        subprocess.run(base+['-pix_fmt', 'yuv420p', str(OUT/f'media/{name}.mp4')], check=True)
        subprocess.run(base+['-vf', 'fps=12.5,scale=360:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                             str(OUT/f'media/{name}.gif')], check=True)
        raw.unlink()


if __name__ == '__main__':
    {'paths': paths, 'odom': odom, 'render': render}[sys.argv[1]]()
