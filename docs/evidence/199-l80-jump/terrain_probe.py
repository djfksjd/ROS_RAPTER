"""Existing l80 run policy at single ledges/drops, with 1 ms contact and current evidence."""
import hashlib
import json
import time
from unittest.mock import patch
import mujoco
import numpy as np
from jump_probe import E, RUN, Recorder, make, loads, save_trace, intervals, reset_stationary
import terrain as tr
from evaluate import load

EDGE = 2.

def add_edge(spec, kind, level, rng, *, height):
    n = int(tr.SIZE/tr.RES)+1
    x = np.linspace(-tr.SIZE/2, tr.SIZE/2, n)
    h = np.zeros((n, n)); h[:, x >= EDGE] = height
    if height < 0:
        spec.worldbody.add_geom(name='edge_pad', type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[(EDGE+tr.SIZE/2)/2, tr.SIZE/2, .05], pos=[(EDGE-tr.SIZE/2)/2, 0, -.05])
        spec.geom('floor').pos = [0, 0, height]
    else:
        spec.worldbody.add_geom(name='edge_ledge', type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[(tr.SIZE/2-EDGE)/2, tr.SIZE/2, height/2], pos=[(EDGE+tr.SIZE/2)/2, 0, height/2])
    return h


class EdgeRecorder(Recorder):
    def __init__(self, env):
        super().__init__(env)
        self.rows['max_foot_x'] = []
    def sample(self, env, dt):
        super().sample(env, dt)
        self.rows['max_foot_x'].append(max(float(env.data.geom_xpos[g, 0]) for gs in env.foot_geoms.values() for g in gs))


def main():
    start = time.process_time()
    out = E/'eval/terrain.jsonl'
    if out.exists(): raise SystemExit('Refusing overwrite')
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (RUN/'model.zip', RUN/'vecnorm.pkl', RUN/'args.json')}
    with out.open('x') as log:
        for height in [.05, .10, -.05, -.10]:
            for seed in range(9200, 9203):
                def builder(spec, kind, level, rng):
                    return add_edge(spec, kind, level, rng, height=height)
                with patch.object(tr, 'add_terrain', builder):
                    env = make(seed, terrain={'kinds': {'boxes': 1}, 'max_level': 0., 'size': 40.})
                    model, vn = load(str(RUN/'model.zip'), env)
                    reset_stationary(env, seed)
                assert env.kind == 'boxes'
                assert np.isclose(tr.height_at(env.heights, 3., 0.), height)
                env.command = np.array([4., 0., 0.]); env.resample_steps = 0
                obs = vn.normalize_obs(env._obs()[None])
                fallen = None; cause = None
                with EdgeRecorder(env) as rec:
                    for k in range(500):
                        raw, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0][0])
                        obs = vn.normalize_obs(raw[None])
                        if term or rec.rows['body_hit'][-20:].count(True):
                            fallen = float(env.data.time)
                            cause = 'body_contact' if rec.rows['body_hit'][-20:].count(True) else 'env_fall'
                            break
                        if env.data.xpos[env.base, 0] > 6: break
                a = rec.arrays()
                reached = float(a['xyz'][-1, 0]) > 6 and fallen is None
                crossing = np.flatnonzero(a['max_foot_x'] > EDGE)
                t_edge = float(a['t'][crossing[0]]) if len(crossing) else None
                window = (a['t'] >= t_edge) & (a['t'] < t_edge+1.) if t_edge is not None else np.zeros(len(a['t']), bool)
                feet_force = a['feet_force_n'].sum(axis=1)
                flights = [(stop-beg)*.001 for beg, stop in intervals((a['feet_n'] == 0).all(axis=1)) if t_edge is not None and a['t'][beg] >= t_edge]
                row = dict(height_m=height, seed=seed, command_mps=4., reached_6m=reached, t_fall=fallen, cause=cause,
                    x_final_m=float(a['xyz'][-1, 0]), duration_s=float(a['t'][-1]), t_edge_s=t_edge,
                    impact_1s_peak_n=float(feet_force[window].max()) if window.any() else None,
                    impact_1s_peak_bw=float(feet_force[window].max()/env.weight) if window.any() else None,
                    current_1s_over_peak=float((a['current_a'][window]/a['current_peak_a'][window]).max()) if window.any() else None,
                    max_flight_after_edge_s=max(flights, default=0), loads=loads(env, a), cpu_s=time.process_time()-start)
                filename = f"terrain_{'up' if height > 0 else 'drop'}_{abs(height):.2f}_{seed}.npz"
                save_trace(E/'eval'/filename, a, env)
                log.write(json.dumps(row)+'\n'); log.flush()
                print(json.dumps({k:v for k,v in row.items() if k != 'loads'}), flush=True)
                vn.close()
    after = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (RUN/'model.zip', RUN/'vecnorm.pkl', RUN/'args.json')}
    assert hashes == after
    (E/'eval/terrain_audit.json').write_text(json.dumps(dict(cpu_s=time.process_time()-start, input_sha256=hashes, hashes_unchanged=True), indent=2)+'\n')

if __name__ == '__main__': main()
