from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load
from evaluate_run import episode,turn_pass
class Adapter:
 def __init__(self,model,offset):self.model,self.offset=model,offset
 def predict(self,obs,**kwargs):
  obs=np.array(obs,copy=True)
  assert obs.shape[1]==46
  mask=obs[:,8]!=0
  assert np.all(obs[mask,6]==np.float32(.4))
  obs[mask,6]+=np.float32(.1*self.offset)
  return self.model.predict(obs,**kwargs)
run=Path('sim/rl/runs/kl005_update64k_20261009');kw=env_kwargs(json.loads((run/'args.json').read_text()));out=Path('sim/rl/runs/screen_logs/turn_speed_feedforward.v2.diagnostic.jsonl')
with out.open('x') as f:
 for offset in [0.,.8]:
  for yaw in [None,-1.,-.5,.5,1.]:
   for seed in [71001,71002]:
    env=RunEnv(**{**kw,'episode_s':15.,'level':0.},randomize=False,seed=7);env.init_seed=seed
    model,norm=load(str(run/'model.zip'),env);assert not norm.norm_obs
    try:
     result=episode(Adapter(model,offset),norm,env,[4.,0.,0.],10.,turn=(yaw,4.,2.) if yaw is not None else None)
     f.write(json.dumps(dict(run=str(run),offset=offset,yaw=yaw,seed=seed,result=result,angle_pass=turn_pass(result,2*yaw) if yaw is not None else None),allow_nan=False)+'\n');f.flush()
    finally:norm.close()
