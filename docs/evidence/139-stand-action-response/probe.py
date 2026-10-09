from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs

def trial(seed,axis,delta):
 cfg=json.loads(Path('/tmp/raptor_stand_knee_170/args.json').read_text());cfg['springs_override']=json.dumps({'knee_pitch':[170,1.5,'bi','latch']})
 e=RunEnv(**{**env_kwargs(cfg),'episode_s':5.,'level':0.},randomize=False,seed=7);e.init_seed=seed;e.reset();e.command=np.zeros(3);e.resample_steps=0;before=[]
 try:
  for i in range(10):
   e.step(np.zeros(len(e.policy_idx)));before.append(dict(qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist()))
  def state():
   w,v=e.body_velocity(e.base);raw=e._obs();return dict(angular_velocity_body=w.tolist(),velocity_body=v.tolist(),gravity_body=raw[3:6].tolist(),qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist())
  start=state();action=np.zeros(len(e.policy_idx))
  for j,k in enumerate(e.policy_idx):
   if e.active[k].endswith('hip_'+axis+'_joint'):action[j]=delta/e.scale[k]
  assert np.max(np.abs(action))<=1
  _,_,fell,_,_=e.step(action);end=state()
  return dict(seed=seed,axis=axis,delta_rad=delta,start=start,end=end,fell=bool(fell),prehistory=before,action=action.tolist(),spring_rest_angle=1.5,knee_design_stiffness=170)
 finally:e.close()
out=Path('sim/rl/runs/screen_logs/stand_action_response.diagnostic.jsonl')
with out.open('x') as f:
 for seed in [22001,22002]:
  control=trial(seed,'pitch',0.)
  for axis,delta in [('pitch',0.),('pitch',-.05),('pitch',.05),('roll',-.05),('roll',.05)]:
   r=control if delta==0 else trial(seed,axis,delta);assert r['prehistory']==control['prehistory'] and r['start']==control['start'];r['exact_start_parity']=True
   f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
   print(seed,axis,delta,r['end']['angular_velocity_body'],flush=True)
