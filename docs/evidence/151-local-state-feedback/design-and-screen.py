from pathlib import Path
import sys,json
import numpy as np
from scipy.linalg import solve_discrete_are
sys.path.insert(0,'/tmp')
from raptor_local_linearization import LocalPlant
sys.path.insert(0,str(Path('sim/rl').resolve()))
from motion_transition import stand_gate
p=Path('sim/rl/runs/screen_logs/local_equilibrium_linearization');mat=np.load(p/'epsilon_1e-06.npz');A,B=mat['A'],mat['B'];plant=LocalPlant()
try:
 blocks=[];block=B.copy()
 for _ in range(plant.nx):blocks.append(block);block=A@block
 U,s,_=np.linalg.svd(np.concatenate(blocks,axis=1),full_matrices=False);rank=int(np.sum(s>s[0]*1e-10));T=U[:,:rank];N=U[:,rank:];Ac,Bc=T.T@A@T,T.T@B
 Q=np.eye(plant.nx);Q[:3,:3]=np.diag([0.,0.,10.]);Q[3:6,3:6]=np.diag([50.,50.,10.]);Q[plant.nv:2*plant.nv,plant.nv:2*plant.nv]*=.1;Q[plant.nv:plant.nv+6,plant.nv:plant.nv+6]=np.diag([2.,2.,2.,5.,5.,1.]);Q[2*plant.nv:,2*plant.nv:]*=.01;R=.1*np.eye(plant.na)
 P=solve_discrete_are(Ac,Bc,T.T@Q@T,R);K=np.linalg.solve(R+Bc.T@P@Bc,Bc.T@P@Ac)@T.T
 closed=np.linalg.eigvals(Ac-Bc@K@T);uncontrolled=np.linalg.eigvals(N.T@A@N) if N.size else np.array([])
 design=dict(reachable_rank=rank,nx=plant.nx,controllability_singular_values=s.tolist(),invariant_subspace_error=float(np.linalg.norm((np.eye(plant.nx)-T@T.T)@A@T)),closed_reachable_spectral_radius=float(max(abs(closed))),unreachable_spectral_radius=float(max(abs(uncontrolled))) if len(uncontrolled) else None,full_simulator_state_feedback=True)
 np.savez(p/'lqr_design.npz',K=K,Q=Q,R=R,T=T,P=P);(p/'lqr_design.json').write_text(json.dumps(design,indent=2)+'\n');print(design,flush=True)
 out=Path('sim/rl/runs/screen_logs/local_lqr_rollout.diagnostic.jsonl')
 with out.open('x') as f:
  for seed in [None,84001,84002]:
   x=np.zeros(plant.nx)
   if seed is not None:
    rng=np.random.default_rng(seed);x[plant.env.v_adr[plant.env.policy_idx]]=rng.normal(0,.001,plant.na);x[plant.nv]=rng.uniform(0,.02)
   plant.restore(x);e=plant.env;rows=[]
   for i in range(500):
    state=plant.state();action=np.clip(plant.u-K@state,-1.,1.);before=e.data.xpos[e.base][:2].copy();_,_,fell,_,info=e.step(action)
    rows.append(dict(t=(i+1)*.02,position_xy=e.data.xpos[e.base][:2].tolist(),before_position_xy=before.tolist(),tilt=float(info['tilt']),flight=bool(info['flight']),speed_xy=float(np.linalg.norm(info['v_body'][:2])),height=float(e.data.xpos[e.base][2]),heat=e.heat_inst.tolist(),action=action.tolist()))
    if fell:break
   heat=np.mean([r['heat'] for r in rows[-2000:]],axis=0);gate=stand_gate(rows);record=dict(seed=seed,seconds=rows[-1]['t'],fell=bool(fell),gate=gate,max_load_average=float(np.max(heat[e.motor_cont<1e8])),rows=rows,full_simulator_state_feedback=True,hardware_state_availability_unverified=True,original_nominal_start_tested=False,catalogue_pass=False)
   f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();print({k:v for k,v in record.items() if k!='rows'},flush=True)
finally:plant.close()
