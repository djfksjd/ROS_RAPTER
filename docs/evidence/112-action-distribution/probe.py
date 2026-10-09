"""Read-only distribution telemetry on reached deterministic states."""
import sys,json,math,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load

def probe(run,enabled):
 env=RunEnv(**{**env_kwargs(json.loads((Path(run)/'args.json').read_text())),'episode_s':25.,'level':0.},randomize=False,seed=7);env.init_seed=60001
 model,norm=load(str(Path(run)/'model.zip'),env);rows=[];controls=[]
 try:
  norm.reset();env.resample_steps=0
  for i in range(1000):
   t=i*.02;stage='stand' if t<5 else 'run' if t<10 else 'turn' if t<15 else 'stop'
   env.command=np.array([0. if stage in ('stand','stop') else 4.,0.,1. if stage=='turn' else 0.])
   obs=norm.normalize_obs(env._obs()[None]);act,_=model.predict(obs,deterministic=True)
   if enabled:
    with torch.no_grad():
     d=model.policy.get_distribution(torch.as_tensor(obs)).distribution
     mu=d.mean.cpu().numpy()[0];std=d.stddev.cpu().numpy()[0]
     assert np.all(std>0)
     cdf=lambda x: np.array([.5*(1+math.erf(v/math.sqrt(2))) for v in x])
     clipping=1-(cdf((1-mu)/std)-cdf((-1-mu)/std))
     np.testing.assert_allclose(act[0],np.clip(mu,-1,1),atol=1e-7)
    rows.append(dict(stage=stage,t=t,mean=mu.tolist(),std=std.tolist(),clip_probability=clipping.tolist()))
   _,reward,done,_,_=env.step(act[0]);controls.append([act[0].tolist(),env.data.qpos.tolist(),env.data.qvel.tolist(),reward])
   if done:break
  return dict(run=run,seed=60001,fell=bool(done),joints=[env.active[i] for i in env.policy_idx],rows=rows,controls=controls)
 finally:norm.close()
p=Path('docs/evidence/112-action-distribution');p.mkdir(exist_ok=True)
with (p/'raw.jsonl').open('x') as f:
 for name in ('seq_compare_catalogue_20261009','stand_support_control_20261009'):
  run='sim/rl/runs/'+name;a=probe(run,False);b=probe(run,True);assert a['controls']==b['controls'];b.pop('controls');b['telemetry_parity_exact']=True;f.write(json.dumps(b,allow_nan=False)+'\n');f.flush();print(name,len(b['rows']),b['fell'],flush=True)
