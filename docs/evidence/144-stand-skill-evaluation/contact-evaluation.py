from pathlib import Path
import sys,json
import numpy as np
import mujoco
sys.path.insert(0,str(Path('sim/rl').resolve()))
from stand_env import StandEnv
from evaluate_stand import trial
original=StandEnv.step
trace=[]
def recorded(self,action):
 r=original(self,action)
 contacts=[]
 if r[2]:
  for i in range(self.data.ncon):
   c=self.data.contact[i]
   other=c.geom2 if c.geom1 in self.ground else c.geom1 if c.geom2 in self.ground else None
   if other is not None and not any(other in g for g in self.foot_geoms.values()):
    force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,force)
    contacts.append(dict(body=self.model.body(int(self.model.geom_bodyid[other])).name,distance_m=float(c.dist),contact_force_normal_N=float(force[0])))
 trace.append(dict(nonfoot_ground_contacts=contacts,reward=r[1],terms=r[4]['stand_terms'],action=action.tolist(),displacement_m=r[4]['stand_displacement_m']))
 return r
StandEnv.step=recorded
try:
 p=Path('sim/rl/runs/screen_logs/stand_skill_final.contact.jsonl')
 with p.open('x') as f:
  for seed in [83001,83002,83003,83004]:
   trace=[];r=trial('sim/rl/runs/stand_skill_pilot_20261009',seed,60.);r['reward_trace']=trace;r['total_raw_reward']=sum(x['reward'] for x in trace)
   f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();print({k:v for k,v in r.items() if k not in ['rows','hashes','reward_trace']},flush=True)
finally:StandEnv.step=original
