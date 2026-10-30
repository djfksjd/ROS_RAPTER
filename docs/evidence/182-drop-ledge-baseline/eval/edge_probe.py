"""Single-edge probe: a policy walks off a drop or into a raised ledge, no retraining, no impulses (randomize off,
realism from args.json at nominal). argv: run_dir drop|ledge <height m> cmd episodes [seed0=5000] [video.mp4].
drop: ground at z = 0 up to the edge at x = 2 m, then z = -height. ledge: z = 0 up to the edge, then a block of
the given height to the end. Fall = tilt > 0.8 rad, body contact, or hip height above the ground under the feet below
half its start value (as terrain_probe.py, but the ground is taken under the mean foot x so that a hip over the ledge
before the feet climb is not counted as low). Success = x > 6 m within 10 s. A run that neither falls nor reaches 6 m
is 'blocked'. Also reported, within 1 s after the first foot passes the edge and sampled at 50 Hz: the largest total
ground force (body weights) and the largest motor current over the motor's peak current (realism layer; 1 = the
RobStride peak torque)."""
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
d, kind, H, vx, n = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
seed0 = int(sys.argv[6]) if len(sys.argv) > 6 else 5000
video = sys.argv[7] if len(sys.argv) > 7 else None
assert kind in ('drop', 'ledge')
tr.SIZE, EDGE = 40., 2.


def add_edge(spec, *_):
    nn = int(tr.SIZE/tr.RES)+1
    x = np.linspace(-tr.SIZE/2, tr.SIZE/2, nn)
    h = np.zeros((nn, nn))
    world = spec.worldbody
    if kind == 'drop':
        h[:, x >= EDGE] = -H
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[(EDGE+tr.SIZE/2)/2, tr.SIZE/2, .05],
                       pos=[(EDGE-tr.SIZE/2)/2, 0, -.05])
        spec.geom('floor').pos = [0, 0, -H]
    else:
        h[:, x >= EDGE] = H
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[(tr.SIZE/2-EDGE)/2, tr.SIZE/2, H/2],
                       pos=[(EDGE+tr.SIZE/2)/2, 0, H/2])
    return h


run_env.tr.add_terrain = add_edge
secs, x_end = 10., 6.
kw = {**env_kwargs(json.loads((d/'args.json').read_text())), 'level': 0., 'episode_s': secs+5.}
rows = []
for ep in range(n):
    env = RunEnv(**kw, randomize=False, seed=seed0+ep, render_mode='rgb_array' if video and ep == 0 else None)
    env.init_seed = seed0+ep
    frames = [] if video and ep == 0 else None
    model, venv = load(str(d/'model.zip'), env)
    venv.reset(); env.command = np.array([vx, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    dd, b, m = env.data, env.base, env.model
    feet = [g for s in env.foot_geoms for g in env.foot_geoms[s]]
    foot_x = lambda: float(np.mean([dd.geom_xpos[g][0] for g in feet]))
    h0 = dd.xpos[b][2]
    fall, cause, vxs, t_edge, grf_max, tm_max = None, None, [], None, 0., 0.
    for t in range(int(secs/.02)):
        raw, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0][0])
        obs = venv.normalize_obs(raw[None])
        x, y, z = dd.xpos[b]
        if t_edge is None and max(dd.geom_xpos[g][0] for g in feet) > EDGE:
            t_edge = t
        if t_edge is not None and t-t_edge < 50:
            grf_max = max(grf_max, sum(info['grf'].values()))
            tm_max = max(tm_max, float(np.max(np.abs(env.rz.current)/env.rz.i_peak)))
        hh = z-tr.height_at(env.heights, foot_x(), y)
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
    xf = float(dd.xpos[b][0])
    rows.append(dict(seed=seed0+ep, t_fall=fall, cause=cause, x=round(xf, 2), blocked=fall is None and xf <= x_end,
                     vx=round(float(np.mean(vxs[50:])), 2) if len(vxs) > 50 else None,
                     grf_max_bw=round(grf_max, 2), current_over_peak_max=round(tm_max, 2)))
print(json.dumps(dict(run=d.name, kind=kind, height=H, cmd=vx, n=n, falls=sum(r['t_fall'] is not None for r in rows),
                      blocked=sum(r['blocked'] for r in rows), episodes=rows)), flush=True)
