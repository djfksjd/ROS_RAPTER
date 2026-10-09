"""Read-only per-mode reward/contact telemetry; no reward recomputation."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from evaluate import load
from train_run import env_kwargs
from run_env import RunEnv,CONTROL_DT
SCHEDULE=[('stand',8.,0.,0.),('walk',6.,.4,0.),('run',8.,4.,0.),('turn_left',4.,4.,1.),('turn_right',4.,4.,-1.),('stop',13.,0.,0.)]
def probe(run,enabled):
 env=RunEnv(**{**env_kwargs(json.loads((Path(run)/'args.json').read_text())),'episode_s':50.,'level':0.},randomize=False,seed=7);env.init_seed=53001
 model,norm=load(str(Path(run)/'model.zip'),env);controls=[];rows=[]
 try:
  norm.reset();env.resample_steps=0;elapsed=0
  for stage,duration,vx,yaw in SCHEDULE:
   env.command=np.array([vx,0.,yaw]);start=elapsed
   for _ in range(round(duration/CONTROL_DT)):
    action,_=model.predict(norm.normalize_obs(env._obs()[None]),deterministic=True)
    _,reward,done,trunc,info=env.step(action[0]);elapsed+=CONTROL_DT
    controls.append([action[0].tolist(),env.data.qpos.tolist(),env.data.qvel.tolist(),reward])
    if enabled:
     weighted={k:float(env.weights[k]*v) for k,v in info['terms'].items()}
     assert np.isclose(reward,CONTROL_DT*sum(weighted.values())-(10. if done else 0.),atol=1e-12)
     rows.append(dict(stage=stage,time=float(env.data.time),stage_time=elapsed-start,command=info['command'].tolist(),reward=reward,weighted_per_second=weighted,terms=info['terms'],v_body=info['v_body'].tolist(),loaded={k:bool(v) for k,v in info['loaded'].items()},phase=float(info['phase']),position=env.data.xpos[env.base,:2].tolist(),flight=bool(info['flight']),tilt=float(info['tilt'])))
    if done or trunc:break
   if done or trunc:break
  return dict(run=run,seed=53001,fell=bool(done),rows=rows,controls=controls)
 finally:norm.close()
p=Path('docs/evidence/108-mode-reward-distribution');p.mkdir(exist_ok=True)
with (p/'raw.jsonl').open('x') as f:
 for kind in ('control','contact'):
  run=f'sim/rl/runs/contact_obs_train_{kind}_20261009';a=probe(run,False);b=probe(run,True);assert a['controls']==b['controls'];b['telemetry_parity_exact']=True;b.pop('controls');f.write(json.dumps(b,allow_nan=False)+'\n');f.flush()
  print(json.dumps(dict(kind=kind,fell=b['fell'],steps=len(b['rows']),parity=True)),flush=True)
