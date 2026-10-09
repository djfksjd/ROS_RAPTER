"""Constant target stand diagnosis in a reconstructed RunEnv; no policy training.

Keep physical model, nominal offsets, coupling, springs and actuator limits.
Target offsets are policy inputs, not changed reset poses or a passive stand claim.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from run_env import RunEnv
from train_run import env_kwargs
from motion_transition import stand_gate


def probe(run,hip_offset,knee_offset,seed,seconds=10.):
    args=json.loads((Path(run)/'args.json').read_text())
    env=RunEnv(**{**env_kwargs(args),'level':0.,'episode_s':seconds+5},randomize=False,seed=7)
    env.init_seed=seed;env.reset();env.command=np.zeros(3);env.resample_steps=0
    action=np.zeros(len(env.policy_idx))
    for j,i in enumerate(env.policy_idx):
        name=env.active[i]
        delta=hip_offset if name.endswith('hip_pitch_joint') else knee_offset if name.endswith('knee_pitch_joint') else 0.
        if not env.lo[i]<=env.q0[i]+delta<=env.hi[i]:raise ValueError('target outside joint range')
        action[j]=delta/env.scale[i]
    if np.max(np.abs(action))>1:raise ValueError('target outside action range')
    rows=[];fell=False
    for k in range(int(seconds/.02)):
        before=env.data.xpos[env.base][:2].copy()
        _,_,fell,_,info=env.step(action)
        rows.append(dict(t=(k+1)*.02,position_xy=env.data.xpos[env.base][:2].tolist(),
                         before_position_xy=before.tolist(),tilt=float(info['tilt']),flight=bool(info['flight']),
                         speed_xy=float(np.linalg.norm(info['v_body'][:2])),height=float(env.data.xpos[env.base][2]),
                         heat=env.heat_inst.tolist()))
        if fell:break
    heat=np.mean([r['heat'] for r in rows[-250:]],axis=0)
    max_heat=float(np.max(heat[env.motor_cont<1e8]));gate=stand_gate(rows)
    return dict(run=str(run),hip_offset=hip_offset,knee_offset=knee_offset,seed=seed,seconds=rows[-1]['t'],
                fell=bool(fell),stand_gate=gate,max_heat=max_heat,action=action.tolist(),
                final_qpos=env.data.qpos[env.q_adr].tolist(),rows=rows,
                pass_gate=not fell and len(rows)==int(seconds/.02) and gate['pass_gate'] and max_heat<=1.,
                physical_temperature=False,passive_stand=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--out',required=True)
    p.add_argument('--hip',type=float,nargs='+',default=[-.2,-.1,0,.1,.2])
    p.add_argument('--knee',type=float,nargs='+',default=[-.4,-.2,0,.2,.4])
    p.add_argument('--seeds',type=int,nargs='+',default=[22001]);p.add_argument('--seconds',type=float,default=10)
    a=p.parse_args()
    if not np.isfinite(a.seconds) or a.seconds<=0:p.error('duration must be finite and positive')
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        for hip in a.hip:
            for knee in a.knee:
                for seed in a.seeds:
                    r=probe(a.run,hip,knee,seed,a.seconds);f.write(json.dumps(r)+'\n');f.flush()
                    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','final_qpos')}),flush=True)

if __name__=='__main__':main()
