from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
from stand_observer import StandObserverController
from motion_transition import stand_gate
source=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009');candidate=next(x for x in map(json.loads,Path('docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1);model=LocalStandModel(source,candidate);model.set_stand_current_limit(.995);design=np.load('sim/rl/runs/screen_logs/local_equilibrium_linearization/observer_design.npz');controller=StandObserverController(design)
try:
 with Path('sim/rl/runs/screen_logs/stand_observer_long.diagnostic.jsonl').open('x') as f:
  for seed in range(85001,85021):
   x=np.zeros(model.nx)
   if seed is not None:
    rng=np.random.default_rng(seed);x[model.env.v_adr[model.env.policy_idx]]=rng.normal(0,.001,8);x[model.nv]=rng.uniform(0,.02)
   model.restore(x);e=model.env;controller.reset();observation=e._obs();rows=[]
   for i in range(3000):
    action=controller.act(observation);before=e.data.xpos[e.base][:2].copy();observation,_,fell,_,info=e.step(action)
    # Ground truth below is logged only after the decision; it is not fed back.
    error=float(np.linalg.norm(design['T']@controller.prior-model.state()))
    rows.append(dict(t=(i+1)*.02,position_xy=e.data.xpos[e.base][:2].tolist(),before_position_xy=before.tolist(),tilt=float(info['tilt']),flight=bool(info['flight']),speed_xy=float(np.linalg.norm(info['v_body'][:2])),height=float(e.data.xpos[e.base][2]),heat=e.heat_inst.tolist(),action=action.tolist(),audit_raw_state_estimation_error=error))
    if fell:break
   heat=np.mean([r['heat'] for r in rows[-2000:]],axis=0);gate=stand_gate(rows);r=dict(near_pose_stand_pass=bool(not fell and len(rows)==3000 and gate["pass_gate"] and max(heat[e.motor_cont<1e8])<=1.),seed=seed,seconds=rows[-1]['t'],fell=bool(fell),gate=gate,max_load_average=float(np.max(heat[e.motor_cont<1e8])),runtime_observation_size=46,passive_joint_ground_truth_used_for_control=False,estimator_initial_truth_injected=False,original_nominal_start_tested=False,catalogue_pass=False,rows=rows)
   f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();print({k:v for k,v in r.items() if k!='rows'},flush=True)
finally:model.close()
