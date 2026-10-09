"""Shadow actor branch telemetry; saved policy outputs and simulator are unchanged."""
import argparse,json
from pathlib import Path
import torch
import numpy as np
from evaluate import load
from run_env import RunEnv
from train_run import env_kwargs


def clipping_flags(a,b):
    a,b=map(np.asarray,(a,b))
    both=(np.abs(a)>1)&(np.abs(b)>1)
    return both,both&(a*b<0)


class ShadowActor:
    def __init__(self,model,env,enabled):self.model,self.env,self.enabled,self.rows=model,env,enabled,[]
    def predict(self,obs,deterministic=True):
        action,state=self.model.predict(obs,deterministic=deterministic)
        if self.enabled:
            p=self.model.policy
            raw=torch.as_tensor(self.env._obs()[None],dtype=torch.float32)
            scaled=raw.clone();scaled[...,8]*=p.yaw_scale
            def branch(x):
                f=p.extract_features(p._normalize(x),p.pi_features_extractor)
                return p.action_net(p.mlp_extractor.forward_actor(f))
            with torch.no_grad():
                a=branch(scaled).cpu().numpy()[0]
                b=(branch(p.mirror(scaled).float()).double()[...,p.reflection_action_index]*p.reflection_action_sign).cpu().numpy()[0]
            both,cancel=clipping_flags(a,b)
            self.rows.append(dict(t=round(float(self.env.data.time),12),raw_branch=a.tolist(),mapped_mirror_branch=b.tolist(),both_clipped=both.tolist(),both_clipped_opposite=cancel.tolist(),action=action[0].tolist()))
        return action,state


def probe(run,yaw,seed,enabled):
    kw=env_kwargs(json.loads((Path(run)/'args.json').read_text()))
    env=RunEnv(**{**kw,'episode_s':15.,'level':0.},randomize=False,seed=7);env.init_seed=seed
    model,norm=load(str(Path(run)/'model.zip'),env);shadow=ShadowActor(model,env,enabled);controls=[]
    try:
        norm.reset();env.command=np.array([4.,0.,0.]);env.resample_steps=0
        if not hasattr(model.policy,'reflection_index'):raise ValueError('requires saved SymmetricPolicy')
        names=[env.active[i] for i in env.policy_idx]
        for _ in range(500):
            t=round(float(env.data.time),12);env.command[2]=yaw if 4.<=t<6. else 0.
            obs=norm.normalize_obs(env._obs()[None]);act,_=shadow.predict(obs)
            _,_,term,_,_=env.step(act[0])
            controls.append(dict(action=act[0].tolist(),qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist()))
            if term:break
        rows=[r for r in shadow.rows if 4.<=r['t']<6.];summary=[]
        if rows:
            both=np.array([r['both_clipped'] for r in rows]);cancel=np.array([r['both_clipped_opposite'] for r in rows])
            for i,n in enumerate(names):summary.append(dict(joint=n,both_clipped_fraction=float(np.mean(both[:,i])),opposite_clipped_fraction=float(np.mean(cancel[:,i]))))
        return dict(run=run,yaw=yaw,seed=seed,fell=bool(term),turn_samples=len(rows),summary=summary,rows=shadow.rows,controls=controls)
    finally:norm.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=45001);a=p.parse_args();out=Path(a.out);out.parent.mkdir(exist_ok=True,parents=True)
    with out.open('x') as f:
        for yaw in [-1.,0.,1.]:
            plain=probe(a.run,yaw,a.seed,False);shadow=probe(a.run,yaw,a.seed,True)
            assert plain['controls']==shadow['controls'],'shadow branch computations changed trajectory'
            shadow['control_parity_exact']=True;f.write(json.dumps(shadow,allow_nan=False)+'\n');f.flush()
            print(json.dumps({k:v for k,v in shadow.items() if k not in ['rows','controls']}),flush=True)

if __name__=='__main__':main()
