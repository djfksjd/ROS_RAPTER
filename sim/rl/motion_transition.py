"""Read-only T1 catalogue baseline: stand and stand/walk/run/stop transitions.

Reconstruct the saved run exactly; retain each failed or incomplete stage.
STOP here is a zero velocity request, not ROS cancellation or power removal.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs


def command_at(t, mode):
    if mode == 'stand':
        return 0., 'stand', 0
    cycle = int(t//34.)
    u = t-cycle*34.
    if u < 5.:
        return 0., 'stand', cycle
    if u < 10.:
        return .4, 'walk', cycle
    if u < 13.:
        return .4+3.6*(u-10.)/3., 'accelerate', cycle
    if u < 21.:
        return 4., 'run', cycle
    return 0., 'stop', cycle


def stand_gate(rows):
    if not rows:
        return dict(pass_gate=False, samples=0)
    pos = np.array([x['position_xy'] for x in rows])
    origin = np.array(rows[0].get('before_position_xy', rows[0]['position_xy']))
    drift = float(np.max(np.linalg.norm(pos-origin, axis=1)))
    tilt = max(x['tilt'] for x in rows)
    flight = sum(x['flight'] for x in rows)
    return dict(pass_gate=bool(drift <= .1 and tilt <= np.deg2rad(10.) and flight == 0),
                samples=len(rows), max_displacement_m=drift, max_tilt_rad=tilt,
                flight_samples=flight)


def alternating_contacts(rows):
    events = []
    for side in ['left','right']:
        i = 0
        while i < len(rows):
            if rows[i]['loaded'][side]:
                j = i
                while j < len(rows) and rows[j]['loaded'][side]:
                    j += 1
                if i > 0 and j-i >= 3:
                    events.append((rows[i]['t'],side))
                i = j
            else:
                i += 1
    events.sort()
    return bool(len(events) >= 4 and all(events[i][1] != events[i-1][1]
                                       and events[i][0] != events[i-1][0]
                                       for i in range(1,len(events))))


def summarize(rows, mode, fell, seconds):
    full = bool(rows and rows[-1]['t'] >= seconds-1e-8 and not fell)
    if mode == 'stand':
        gate = stand_gate(rows)
        return dict(completed=full, pass_gate=bool(full and gate['pass_gate']), stand=gate)
    cycles = []
    for cycle in [0,1]:
        rr = [x for x in rows if x['cycle'] == cycle]
        initial_stand = [x for x in rr if x['stage']=='stand']
        walk = [x for x in rr if x['stage']=='walk']
        run = [x for x in rr if x['stage']=='run']
        stop = [x for x in rr if x['stage']=='stop']
        held = [x for x in stop if x['t'] >= 34.*cycle+24.]
        stop_time, count = None, 0
        for x in stop:
            count = count+1 if x['speed_xy'] < .1 else 0
            if count == 25 and stop_time is None:
                stop_time = x['t']-(34.*cycle+21.)-.5
        run_speed = float(np.mean([x['vx'] for x in run[-200:]])) if run else None
        start_speed = run[-1]['speed_xy'] if run else None
        standing = stand_gate(held)
        stand_start = stand_gate(initial_stand)
        walk_speed = float(np.mean([x['vx'] for x in walk[-125:]])) if walk else None
        walk_pass = bool(len(walk) >= 250 and walk_speed is not None and abs(walk_speed-.4) <= .1
                         and alternating_contacts(walk))
        hold_complete = len(held) >= 500
        stop_gate = bool(start_speed is not None and start_speed >= 3.6
                         and stop_time is not None and stop_time <= 3.
                         and hold_complete and standing['pass_gate']
                         and all(x['speed_xy'] < .1 for x in held))
        pos = np.array([x['position_xy'] for x in stop])
        cycles.append(dict(cycle=cycle, run_speed_mps=run_speed,
                           initial_stand=stand_start,
                           initial_stand_pass=bool(len(initial_stand) >= 250 and stand_start['pass_gate']),
                           walk_speed_mps=walk_speed, walk_pass=walk_pass,
                           stop_start_speed_mps=start_speed, stop_time_s=stop_time,
                           distance_after_stop_m=float(np.linalg.norm(pos[-1]-stop[0]['before_position_xy'])) if len(pos) else None,
                           hold_complete=hold_complete, stand_after_stop=standing,
                           stop_pass=stop_gate,
                           run_pass=bool(run_speed is not None and abs(run_speed-4.) <= .4)))
    return dict(completed=full, pass_gate=bool(full and all(x['initial_stand_pass'] and x['walk_pass']
                                                         and x['stop_pass'] and x['run_pass'] for x in cycles)),
                cycles=cycles)


def probe(run, seed, mode, yaw_scale=None):
    seconds = 60. if mode == 'stand' else 68.
    args = json.loads((Path(run)/'args.json').read_text())
    env = RunEnv(**{**env_kwargs(args), 'level':0., 'episode_s':seconds+5.}, randomize=False, seed=7)
    env.init_seed = seed
    model, venv = load(str(Path(run)/'model.zip'), env)
    rows = []
    try:
        venv.reset(); env.resample_steps = 0
        if yaw_scale is not None:
            from symmetric_rollout import RawPolicyAdapter
            model=RawPolicyAdapter(model,venv,env,yaw_scale)
        for k in range(int(seconds/.02)):
            t = k*.02
            speed, stage, cycle = command_at(t, mode)
            env.command = np.array([speed,0.,0.])
            obs = venv.normalize_obs(env._obs()[None])
            action, _ = model.predict(obs, deterministic=True)
            before_position = env.data.xpos[env.base][:2].copy()
            _, _, fell, _, info = env.step(action[0])
            rows.append(dict(t=(k+1)*.02, command_vx=speed, stage=stage, cycle=cycle,
                             vx=float(info['v_body'][0]), speed_xy=float(np.linalg.norm(info['v_body'][:2])),
                             position_xy=env.data.xpos[env.base][:2].tolist(), tilt=float(info['tilt']),
                             before_position_xy=before_position.tolist(),
                             loaded={s:bool(v) for s,v in info['loaded'].items()},
                             flight=bool(info['flight']), heat=env.heat_inst.tolist()))
            if fell:
                break
        return dict(run=run, seed=seed, mode=mode, fell=bool(fell), actual_seconds=rows[-1]['t'],yaw_scale=yaw_scale,
                    summary=summarize(rows,mode,bool(fell),seconds), rows=rows)
    finally:
        venv.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+')
    p.add_argument('--out',required=True)
    p.add_argument('--yaw-scale',type=float)
    p.add_argument('--modes',nargs='+',choices=['stand','transition'],default=['stand','transition'])
    p.add_argument('--seeds',nargs='+',type=int,default=[12001,12002])
    a = p.parse_args()
    if a.yaw_scale is not None and (not np.isfinite(a.yaw_scale) or a.yaw_scale<=0):
        p.error("yaw scale must be finite and positive")
    out = Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        p.error(f'refusing to overwrite {out}')
    with out.open('x') as f:
        for run in a.runs:
            for mode in a.modes:
                for seed in a.seeds:
                    result = probe(run,seed,mode,a.yaw_scale)
                    f.write(json.dumps(result)+'\n'); f.flush()
                    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()
