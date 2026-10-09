from pathlib import Path
import sys,json
import numpy as np
from scipy.linalg import solve_discrete_are
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
source=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009')
candidate=next(x for x in map(json.loads,Path('docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1)
model=LocalStandModel(source,candidate);folder=Path('sim/rl/runs/screen_logs/local_equilibrium_linearization');matrix=np.load(folder/'epsilon_1e-06.npz');design=np.load(folder/'lqr_design.npz');T=design['T'];A,B=matrix['A'],matrix['B'];Ac,Bc=T.T@A@T,T.T@B;indices=np.r_[np.arange(6),np.arange(9,44)];zero=np.zeros(model.nx)
try:
 model.restore(zero);reference=model.env._obs().astype(float);assert reference.shape==(46,)
 C=np.empty((len(indices),model.nx));epsilon=1e-4
 for i in range(model.nx):
  delta=zero.copy();delta[i]=epsilon;model.restore(delta);plus=model.env._obs()[indices].astype(float);model.restore(-delta);minus=model.env._obs()[indices].astype(float);C[:,i]=(plus-minus)/(2*epsilon)
 Cc=C@T;W=1e-7*np.eye(Ac.shape[0]);V=1e-6*np.eye(len(indices));P=solve_discrete_are(Ac.T,Cc.T,W,V);L=np.linalg.solve(Cc@P@Cc.T+V,Cc@P).T
 eig=np.linalg.eigvals(Ac);unstable=[]
 for value in eig:
  if abs(value)>1.0001:
   sv=np.linalg.svd(np.r_[value*np.eye(Ac.shape[0])-Ac,Cc],compute_uv=False);unstable.append(dict(eigenvalue=float(value.real),observability_pbh_min_singular=float(sv[-1])))
 error_eig=np.linalg.eigvals(Ac@(np.eye(len(Ac))-L@Cc))
 out=folder/'observer_design.npz';np.savez(out,Ac=Ac,Bc=Bc,Cc=Cc,L=L,T=T,Kc=design['K']@T,reference=reference,indices=indices,u_reference=model.u,affine=T.T@matrix['baseline_step'],C_full=C,process_covariance=W,sensor_covariance=V)
 result=dict(measured_original_observation_size=46,selected_measurements=len(indices),finite_difference=epsilon,unstable_observable_modes=unstable,observer_error_spectral_radius=float(max(abs(error_eig))),controller_runtime_requires_passive_joint_measurements=False,covariances_are_design_assumptions=True,measurement_pipeline_has_existing_mixed_timestamp_caches=True)
 (folder/'observer_design.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
finally:model.close()
