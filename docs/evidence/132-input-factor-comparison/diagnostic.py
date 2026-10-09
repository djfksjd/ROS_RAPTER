from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load
from evaluate_run import episode,turn_pass
class MeasuredEnv(RunEnv):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.turn_vx=[]
 def step(self,a):
  t=round(float(self.data.time),12);r=super().step(a)
  if 4.<=t<6.:self.turn_vx.append(float(r[4]['v_body'][0]))
  return r
class Adapter:
 def __init__(self,model,yaw_factor,speed_feedback):self.model,self.yaw_factor,self.speed_feedback=model,yaw_factor,speed_feedback
 def predict(self,obs,**kw):
  x=np.array(obs,copy=True);assert x.shape[1]==46 and np.isfinite(x).all();mask=x[:,8]!=0;assert np.all(x[mask,6]==np.float32(.4))
  x[mask,8]*=np.float32(self.yaw_factor)
  if self.speed_feedback:x[mask,6]+=np.asarray(.1*np.clip(.8+1.2*(4.-x[mask,9]*10.),0.,1.5),dtype=np.float32)
  return self.model.predict(x,**kw)
run=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009');kw=env_kwargs(json.loads((run/'args.json').read_text()));out=Path('sim/rl/runs/screen_logs/low_lr_adapter_factors.diagnostic.jsonl')
with out.open('x') as f:
 for yf,feedback in [(1.,False),(1.1,False),(1.,True),(1.1,True)]:
  for yaw in [None,-1.,-.5,.5,1.]:
   for seed in [71001,71002]:
    env=MeasuredEnv(**{**kw,'episode_s':15.,'level':0.},randomize=False,seed=7);env.init_seed=seed;model,norm=load(str(run/'model.zip'),env);assert not norm.norm_obs
    try:
     result=episode(Adapter(model,yf,feedback),norm,env,[4.,0.,0.],10.,turn=(yaw,4.,2.) if yaw is not None else None)
     f.write(json.dumps(dict(run=str(run),yaw_factor=yf,speed_feedback=feedback,yaw=yaw,seed=seed,result=result,angle_pass=turn_pass(result,2*yaw) if yaw is not None else None,measured_interval_max_vx=max(env.turn_vx) if env.turn_vx else None),allow_nan=False)+'\n');f.flush()
    finally:norm.close()
