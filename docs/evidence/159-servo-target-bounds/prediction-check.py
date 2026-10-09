from pathlib import Path
import sys,json,numpy as np,mujoco
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
candidate=next(x for x in map(json.loads,Path('docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1)
p=LocalStandModel('sim/rl/runs/kl005_lr3e5_update64k_20261009',candidate)
design=np.load('sim/rl/runs/screen_logs/local_equilibrium_linearization/epsilon_1e-06.npz');A=design['A'];B=design['B'];f0=design['baseline_step']
rs=[json.loads(x) for x in Path('docs/evidence/158-contact-and-handoff/contact-pair.jsonl').read_text().splitlines()];results=[]
def state_at(r,t):
 snap=min(r['substeps'],key=lambda s:abs(s['state_time']-t));dq=np.zeros(p.nv);mujoco.mj_differentiatePos(p.env.model,dq,1.,p.q,np.array(snap['qpos_after']));action=r['rows'][round(t/.02)-1]['action'];return np.r_[dq,snap['qvel_after'],np.array(action)-p.u]
try:
 for r in rs:
  if not r['telemetry']:continue
  for t in [.58,.6,.62]:
   x=state_at(r,t);actual=state_at(r,t+.02);u=np.array(r['rows'][round((t+.02)/.02)-1]['action'])-p.u
   linear=f0+A@x+B@u
   error=None
   try:nonlinear=p.transition(x,u)
   except ValueError as ex:nonlinear=p.state();error=str(ex)
   denom=max(float(np.linalg.norm(actual-x)),1e-12)
   results.append(dict(seed=r['seed'],start_seconds=t,state_change_norm=float(np.linalg.norm(actual-x)),linear_error_norm=float(np.linalg.norm(linear-actual)),nonlinear_error_norm=float(np.linalg.norm(nonlinear-actual)),linear_relative_to_change=float(np.linalg.norm(linear-actual))/denom,nonlinear_relative_to_change=float(np.linalg.norm(nonlinear-actual))/denom,nonlinear_region_error=error,online_true_state_used=False,offline_initial_truth_used=True))
finally:p.close()
Path('sim/rl/runs/screen_logs/stand_prediction_check.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
