from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs
from motion_transition import stand_gate

def trial(rest,seed,feedback):
 cfg=json.loads(Path('/tmp/raptor_stand_knee_170/args.json').read_text());cfg['springs_override']=json.dumps({'knee_pitch':[170,rest,'bi','latch']})
 e=RunEnv(**{**env_kwargs(cfg),'episode_s':15.,'level':0.},randomize=False,seed=7);e.init_seed=seed;e.reset();e.command=np.zeros(3);e.resample_steps=0;rows=[]
 try:
  for i in range(500):
   raw=e._obs();w,v=e.body_velocity(e.base);pitch=float(np.arctan2(raw[3],-raw[5]));roll=float(np.arctan2(-raw[4],-raw[5]));a=np.zeros(len(e.policy_idx))
   dp=float(np.clip(.25*pitch+.05*w[1],-.25,.25)) if feedback else 0.
   dr=float(np.clip(.20*roll+.04*w[0],-.15,.15)) if feedback else 0.
   for j,k in enumerate(e.policy_idx):
    n=e.active[k];delta=dp if n.endswith('hip_pitch_joint') else dr if n.endswith('hip_roll_joint') else 0.;a[j]=delta/e.scale[k]
   assert np.max(np.abs(a))<=1.
   before=e.data.xpos[e.base][:2].copy();_,_,fell,_,info=e.step(a)
   rows.append(dict(t=(i+1)*.02,position_xy=e.data.xpos[e.base][:2].tolist(),before_position_xy=before.tolist(),tilt=float(info['tilt']),flight=bool(info['flight']),speed_xy=float(np.linalg.norm(info['v_body'][:2])),height=float(e.data.xpos[e.base][2]),heat=e.heat_inst.tolist(),action=a.tolist(),pitch_error=pitch,roll_error=roll))
   if fell:break
  return dict(rest_angle=rest,seed=seed,feedback=feedback,fell=bool(fell),seconds=rows[-1]['t'],stand_gate=stand_gate(rows),rows=rows)
 finally:e.close()
p=Path('sim/rl/runs/screen_logs/stand_orientation_feedback.diagnostic.jsonl')
with p.open('x') as f:
 for rest in [.956,1.5]:
  for feedback in [False,True]:
   for seed in [22001,22002]:
    r=trial(rest,seed,feedback);f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();print({k:v for k,v in r.items() if k!='rows'},flush=True)
