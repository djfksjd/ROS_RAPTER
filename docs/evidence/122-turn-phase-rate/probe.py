from pathlib import Path
import sys,json
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv,CONTROL_DT
from train_run import env_kwargs
from evaluate import load
from evaluate_run import episode,turn_pass
class PhaseRateEnv(RunEnv):
 def __init__(self,*args,turn_boost_hz=0.,**kwargs):super().__init__(*args,**kwargs);self.turn_boost_hz=turn_boost_hz
 def step(self,action):
  result=super().step(action)
  if self.turn_boost_hz and self.command[2]!=0:
   self.phase=(self.phase+CONTROL_DT*self.turn_boost_hz)%1.
   return self._observe(),*result[1:]
  return result
run=Path('sim/rl/runs/kl005_update64k_20261009');kw=env_kwargs(json.loads((run/'args.json').read_text()));out=Path('sim/rl/runs/screen_logs/turn_phase_rate.diagnostic.jsonl')
with out.open('x') as f:
 for hz in [0.,.3]:
  for yaw in [None,-1.,-.5,.5,1.]:
   for seed in [71001,71002]:
    env=PhaseRateEnv(**{**kw,'episode_s':15.,'level':0.},turn_boost_hz=hz,randomize=False,seed=7);env.init_seed=seed
    model,norm=load(str(run/'model.zip'),env)
    try:
     result=episode(model,norm,env,[4.,0.,0.],10.,turn=(yaw,4.,2.) if yaw is not None else None)
     f.write(json.dumps(dict(run=str(run),boost_hz=hz,yaw=yaw,seed=seed,result=result,angle_pass=turn_pass(result,2*yaw) if yaw is not None else None),allow_nan=False)+'\n');f.flush()
    finally:norm.close()
