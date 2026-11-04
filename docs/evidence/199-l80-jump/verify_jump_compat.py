"""Read-only default-rollout regression; raw snapshots remain local, summaries tracked."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'sim/rl'))
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load
E=Path(__file__).resolve().parent

def sample():
    arrays={}
    for name,cmd in [('l80_s1p',0.),('l80_w2',1.),('l80_r1_6M',4.)]:
        d=ROOT/'sim/rl/runs'/name
        for dr in (False,True):
            env=RunEnv(**env_kwargs(json.loads((d/'args.json').read_text())),randomize=dr,seed=8510)
            env.init_seed=8510
            model,vn=load(str(d/'model.zip'),env);vn.reset()
            env.command=np.array([cmd,0.,0.]);env.resample_steps=0
            obs=vn.normalize_obs(env._obs()[None]);rows=[]
            for k in range(150):
                action=model.predict(obs,deterministic=True)[0][0]
                raw,reward,term,trunc,_=env.step(action);obs=vn.normalize_obs(raw[None])
                rows.append(np.r_[raw,action,reward,term,trunc,env.data.qpos,env.data.qvel])
                if term or trunc:break
            arrays[f'{name}_{int(dr)}']=np.array(rows);vn.close()
    return arrays

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');a=p.parse_args()
    path=E/'eval/jump_default_before.npz';out=sample()
    if a.baseline:
        if path.exists():raise SystemExit('baseline exists')
        np.savez_compressed(path,**out)
    else:
        old=np.load(path)
        for k,v in out.items():np.testing.assert_array_equal(v,old[k],err_msg=k)
        (E/'eval/jump_default_compat.json').write_text(json.dumps(dict(exact=True,rollouts={k:len(v) for k,v in out.items()},
            baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
    print({k:len(v) for k,v in out.items()})
if __name__=='__main__':main()
