"""Zero-shot terrain probe: a flat-trained policy on terrain.py ground, no retraining, no impulses (randomize off,
realism from args.json at nominal). argv: run_dir kind value cmd episodes [seed0=5000].
kind/value: flat 0 | rough <bump amplitude m> | slope_up / slope_down <deg> | side_slope <deg> | stairs <rise m>.
The ground is enlarged to 40 m so the robot can walk 19 m ahead. Fall = tilt > 0.8 rad, body contact, or hip height
above the local ground below half its start value (RunEnv's own check uses absolute height, wrong on slopes).
Success = no fall within 10 s or before reaching x = 18 m."""
import json, sys
from pathlib import Path
import mujoco
import numpy as np
sys.path.insert(0, 'sim/rl')
import terrain as tr
import run_env
from evaluate import load, write_mp4
from run_env import RunEnv
from train_run import env_kwargs
d, kind, val, vx, n = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
seed0 = int(sys.argv[6]) if len(sys.argv) > 6 else 5000
video = sys.argv[7] if len(sys.argv) > 7 else None  # optional mp4 of the first episode
tr.SIZE = 40.
base, sign = kind.split('_')[0] if kind.startswith('slope') else kind, (-1. if kind == 'slope_down' else 1.)
level = {'flat': 0., 'rough': (val-.005)/.045, 'slope': (val-3.)/22., 'side_slope': (val-3.)/17., 'stairs': (val-.02)/.10}[base]
_heights, _add = tr.heights, tr.add_terrain
def heights(k, lv, rng):
    h = _heights(k, lv, rng)
    if k == 'slope' and np.sign(h[0, -1]-h[0, h.shape[1]//2]) != sign: h = -h
    return h
tr.heights = heights


def add_rough_boxes(spec, lv, rng):
    """rough as 0.2 m box patches (same height draw as terrain.heights) in a 3 m wide strip x 0.5-19.5 m.
    terrain.py's height-field version lets the foot boxes sink 2-5 cm into the ground (MuJoCo 3.4 box-hfield
    contacts), so a standing policy falls even on a 0.5 cm field; boxes are exact."""
    h = _heights('rough', lv, rng)
    n, x = h.shape[0], np.linspace(-tr.SIZE/2, tr.SIZE/2, h.shape[0])
    lo = min(h.min(), 0.)-.02
    keep = np.zeros_like(h, bool)
    world = spec.worldbody
    for i in range(0, n-3, 4):  # rows = y
        for j in range(0, n-3, 4):  # columns = x
            xc, yc = x[j]+.075, x[i]+.075
            if not (.5 <= x[j] and xc < 19.6 and abs(yc) < 1.5):
                continue
            keep[i:i+4, j:j+4] = True
            top = h[i, j]
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.1, .1, (top-lo)/2], pos=[xc, yc, (top+lo)/2])
    h[~keep] = lo
    x0 = x[np.flatnonzero(keep.any(0))[0]]-.025  # first patch edge: the pad reaches it with no gap
    h[np.tile(x, (n, 1)) < x0] = 0.
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[(tr.SIZE/2+x0)/2, tr.SIZE/2, (0-lo)/2],
                   pos=[(-tr.SIZE/2+x0)/2, 0, lo/2])  # spawn pad and everything behind at z = 0
    spec.geom('floor').pos = [0, 0, lo]
    return h


run_env.tr.add_terrain = lambda spec, k, lv, rng: add_rough_boxes(spec, level, rng) if base == 'rough' else _add(spec, base, level, rng)
secs, x_end = 10., 18.
kw = {**env_kwargs(json.loads((d/'args.json').read_text())), 'level': 0., 'episode_s': secs+5.}
rows = []
for ep in range(n):
    env = RunEnv(**kw, randomize=False, seed=seed0+ep, render_mode='rgb_array' if video and ep == 0 else None)
    env.init_seed = seed0+ep
    frames = [] if video and ep == 0 else None
    model, venv = load(str(d/'model.zip'), env)
    venv.reset(); env.command = np.array([vx, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    dd, b = env.data, env.base
    h0 = dd.xpos[b][2]-tr.height_at(env.heights, *dd.xpos[b][:2])
    fall, cause, vxs = None, None, []
    for t in range(int(secs/.02)):
        raw, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0][0])
        obs = venv.normalize_obs(raw[None])
        x, y, z = dd.xpos[b]
        hh = z-tr.height_at(env.heights, x, y)
        bad = 'tilt' if info['tilt'] > .8 else 'body' if info['terms'].get('collision', 0.) > 0 else 'low' if hh < .5*h0 else \
              'nan' if not np.isfinite(dd.qpos).all() else None
        if bad:
            fall, cause = round((t+1)*.02, 2), bad
            if frames is not None:
                for _ in range(25): env.step(model.predict(obs, deterministic=True)[0][0]); frames.append(env.render())
            break
        vxs.append(info['v_body'][0])
        if frames is not None: frames.append(env.render())
        if x > x_end: break
    if frames is not None:
        for _ in range(25): frames.append(env.render())  # hold the end (fall or finish) for 0.5 s
        write_mp4(video, frames)
    rows.append(dict(seed=seed0+ep, t_fall=fall, cause=cause, x=round(float(dd.xpos[b][0]), 2),
                     vx=round(float(np.mean(vxs[50:])), 2) if len(vxs) > 50 else None))
print(json.dumps(dict(run=d.name, kind=kind, value=val, cmd=vx, n=n, falls=sum(r['t_fall'] is not None for r in rows), episodes=rows)), flush=True)
