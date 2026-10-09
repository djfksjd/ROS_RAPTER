from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,'/tmp')
from raptor_local_linearization import LocalPlant
sys.path.insert(0,str(Path('sim/rl').resolve()))
from motion_transition import stand_gate
from run_env import LINK_ETA
plant=LocalPlant();p=Path('sim/rl/runs/screen_logs/local_equilibrium_linearization');K=np.load(p/'lqr_design.npz')['K']
e=plant.env;original_clamp=e._clamp
def limited_clamp():
 low,high=original_clamp();g=e.gear_const.copy();g[e.fourbar_idx]=np.interp(e.data.qpos[e.q_adr[e.fourbar_idx]],*e.knee_ratio);cap=.995*g*LINK_ETA*e.motor_cont;low=np.maximum(low,-cap);high=np.minimum(high,cap);e.model.actuator_forcerange[e.act,0]=low;e.model.actuator_forcerange[e.act,1]=high;return low,high
e._clamp=limited_clamp
try:
 with Path('sim/rl/runs/screen_logs/local_lqr_cap_all.diagnostic.jsonl').open('x') as f:
  for seed in range(85001,85021):
   rng=np.random.default_rng(seed);x=np.zeros(plant.nx);x[plant.env.v_adr[plant.env.policy_idx]]=rng.normal(0,.001,plant.na);x[plant.nv]=rng.uniform(0,.02)
   plant.restore(x);e=plant.env;rows=[]
   for i in range(3000):
    action=np.clip(plant.u-K@plant.state(),-1.,1.);before=e.data.xpos[e.base][:2].copy();_,_,fell,_,info=e.step(action)
    rows.append(dict(t=(i+1)*.02,position_xy=e.data.xpos[e.base][:2].tolist(),before_position_xy=before.tolist(),tilt=float(info['tilt']),flight=bool(info['flight']),speed_xy=float(np.linalg.norm(info['v_body'][:2])),height=float(e.data.xpos[e.base][2]),heat=e.heat_inst.tolist(),action=action.tolist()))
    if fell:break
   heat=np.mean([r['heat'] for r in rows[-2000:]],axis=0);load=float(np.max(heat[e.motor_cont<1e8]));gate=stand_gate(rows);r=dict(stand_software_continuous_limit_factor=.995,seed=seed,initial_delta=x.tolist(),seconds=rows[-1]['t'],fell=bool(fell),gate=gate,max_load_last40=load,near_pose_stand_pass=bool(not fell and len(rows)==3000 and gate['pass_gate'] and load<=1.),full_simulator_state_feedback=True,hardware_state_availability_unverified=True,original_nominal_start_tested=False,catalogue_pass=False,rows=rows)
   f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();print({k:v for k,v in r.items() if k not in ['rows','initial_delta']},flush=True)
finally:plant.close()
