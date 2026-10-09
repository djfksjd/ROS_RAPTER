"""Isolated policy target-range diagnosis; unchanged joints and actuator limits."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load
from evaluate_run import episode,turn_pass

class Recorder:
    def __init__(self,model,env):self.model,self.env,self.rows=model,env,[]
    def predict(self,obs,deterministic=True):
        action,state=self.model.predict(obs,deterministic=deterministic)
        env=self.env;indices=[i for i in env.policy_idx if env.active[i].endswith('hip_roll_joint')]
        slots=[list(env.policy_idx).index(i) for i in indices]
        self.rows.append(dict(t=float(env.data.time),yaw=float(env.command[2]),
                              action=action[0,slots].tolist(),
                              target=(env.q0[indices]+env.scale[indices]*action[0,slots]).tolist(),
                              actual=env.data.qpos[env.q_adr[indices]].tolist(),
                              torque=env.data.actuator_force[env.act[indices]].tolist()))
        return action,state


def hip_tracking_summary(rows,yaw):
    selected=[x for x in rows if 4.<=round(x['t'],12)<6.] if yaw is not None else rows
    if not selected:
        return {'measured_samples':0,'action_saturation_fraction':None,'hip_tracking_rms_rad':None}
    actions=np.array([x['action'] for x in selected])
    errors=np.array([np.array(x['target'])-x['actual'] for x in selected])
    return {'measured_samples':len(selected),
            'action_saturation_fraction':np.mean(np.abs(actions)>=.98,axis=0).tolist(),
            'hip_tracking_rms_rad':np.sqrt(np.mean(errors**2,axis=0)).tolist()}


def probe(run,scale,speed,yaw,seed):
    args=json.loads((Path(run)/'args.json').read_text())
    env=RunEnv(**{**env_kwargs(args),'level':0.,'episode_s':15.},randomize=False,seed=7)
    indices=[i for i,n in enumerate(env.active) if n.endswith('hip_roll_joint')]
    for i in indices:env.scale[i]=scale
    env.init_seed=seed;model,norm=load(str(Path(run)/'model.zip'),env)
    recorder=Recorder(model,env)
    try:
        r=episode(recorder,norm,env,[speed,0.,0.],10.,turn=(yaw,4.,2.) if yaw is not None else None)
        for i in indices:
            if env.q0[i]-scale<env.lo[i] or env.q0[i]+scale>env.hi[i]:
                raise ValueError('policy target range exceeds mechanical range')
        return dict(run=str(run),scale=scale,speed=speed,yaw=yaw,initial_seed=seed,result=r,
                    turn_pass_20pct=turn_pass(r,2*yaw) if yaw is not None else None,
                    **hip_tracking_summary(recorder.rows,yaw),
                    active_dof=len(env.policy_idx),mechanical_range_unchanged=True,rows=recorder.rows)
    finally:norm.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--out',required=True)
    p.add_argument('--scales',type=float,nargs='+',default=[.3,.45]);p.add_argument('--seeds',type=int,nargs='+',default=[35001,35002,35003]);a=p.parse_args()
    if any(not np.isfinite(s) or s<=0 or s>.5 for s in a.scales):p.error('scales must be finite and within (0,.5]')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:
        for scale in a.scales:
            for speed,yaw in [(4.,None),(7.,None),(4.,-1.),(4.,-.5),(4.,.5),(4.,1.)]:
                for seed in a.seeds:
                    r=probe(a.run,scale,speed,yaw,seed);f.write(json.dumps(r)+'\n');f.flush()
                    print(json.dumps({k:v for k,v in r.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
