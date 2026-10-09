from pathlib import Path
import sys,json,numpy as np,mujoco
from scipy.linalg import solve_discrete_are
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
from motion_transition import stand_gate
candidates=[json.loads(x) for x in Path('sim/rl/runs/screen_logs/static_equilibrium_pitch_family.diagnostic.jsonl').read_text().splitlines()]
folder=Path('sim/rl/runs/screen_logs/pitch_family_design');folder.mkdir(exist_ok=False)
for number,candidate in enumerate(candidates):
 if not candidate['static_feasible']:continue
 plant=LocalStandModel('sim/rl/runs/kl005_lr3e5_update64k_20261009',candidate);out=folder/str(number);out.mkdir();zero=np.zeros(plant.nx)
 try:
  plant.restore(zero);d=plant.env.data;validation=dict(max_root_linear_acc=float(max(abs(d.qacc[:3]))),max_root_angular_acc=float(max(abs(d.qacc[3:6]))),max_joint_acc=float(max(abs(d.qacc[6:]))))
  f0=plant.transition(zero,np.zeros(8));A=np.empty((plant.nx,plant.nx));B=np.empty((plant.nx,8));epsilon=1e-6
  for i in range(plant.nx):
   delta=zero.copy();delta[i]=epsilon;A[:,i]=(plant.transition(delta,np.zeros(8))-plant.transition(-delta,np.zeros(8)))/(2*epsilon)
  for i in range(8):
   delta=np.zeros(8);delta[i]=epsilon;B[:,i]=(plant.transition(zero,delta)-plant.transition(zero,-delta))/(2*epsilon)
  blocks=[];block=B.copy()
  for _ in range(plant.nx):blocks.append(block);block=A@block
  U,s,_=np.linalg.svd(np.concatenate(blocks,axis=1),full_matrices=False);rank=int(np.sum(s>s[0]*1e-10));T=U[:,:rank];Ac,Bc=T.T@A@T,T.T@B
  Q=np.eye(plant.nx);Q[:3,:3]=np.diag([0.,0.,10.]);Q[3:6,3:6]=np.diag([50.,50.,10.]);Q[plant.nv:2*plant.nv,plant.nv:2*plant.nv]*=.1;Q[plant.nv:plant.nv+6,plant.nv:plant.nv+6]=np.diag([2.,2.,2.,5.,5.,1.]);Q[2*plant.nv:,2*plant.nv:]*=.01;R=.1*np.eye(8)
  P=solve_discrete_are(Ac,Bc,T.T@Q@T,R);Kc=np.linalg.solve(R+Bc.T@P@Bc,Bc.T@P@Ac);K=Kc@T.T
  indices=np.r_[np.arange(6),np.arange(9,44)];plant.restore(zero);reference=plant.env._obs().astype(float);C=np.empty((41,plant.nx))
  for i in range(plant.nx):
   delta=zero.copy();delta[i]=1e-4;plant.restore(delta);plus=plant.env._obs()[indices].astype(float);plant.restore(-delta);minus=plant.env._obs()[indices].astype(float);C[:,i]=(plus-minus)/2e-4
  Cc=C@T;W=1e-7*np.eye(rank);V=1e-6*np.eye(41);Po=solve_discrete_are(Ac.T,Cc.T,W,V);L=np.linalg.solve(Cc@Po@Cc.T+V,Cc@Po).T
  np.savez(out/'design.npz',Ac=Ac,Bc=Bc,Cc=Cc,L=L,T=T,K=K,Kc=Kc,reference=reference,indices=indices,u_reference=plant.u,affine=T.T@f0,A=A,B=B,f0=f0,Q=Q,R=R)
  info=dict(number=number,pitch=candidate['fixed_pitch'],validation=validation,f0_norm=float(np.linalg.norm(f0)),reachable_rank=rank,controller_radius=float(max(abs(np.linalg.eigvals(Ac-Bc@Kc)))),observer_radius=float(max(abs(np.linalg.eigvals(Ac@(np.eye(rank)-L@Cc))))),catalogue_pass=False)
  (out/'design.json').write_text(json.dumps(info,indent=2)+'\n');print(info,flush=True)
  plant.set_stand_current_limit(.995)
  with (out/'near-screen.jsonl').open('x') as log:
   for seed in [None,84001,84002]:
    x=zero.copy()
    if seed is not None:
     rng=np.random.default_rng(seed);x[plant.env.v_adr[plant.env.policy_idx]]=rng.normal(0,.001,8);x[plant.nv]=rng.uniform(0,.02)
    plant.restore(x);e=plant.env;rows=[]
    for step in range(500):
     action=np.clip(plant.u-K@plant.state(),-1.,1.);before=e.data.xpos[e.base][:2].copy();_,_,fell,_,r=e.step(action)
     rows.append(dict(t=(step+1)*.02,position_xy=e.data.xpos[e.base][:2].tolist(),before_position_xy=before.tolist(),tilt=float(r['tilt']),flight=bool(r['flight']),speed_xy=float(np.linalg.norm(r['v_body'][:2])),height=float(e.data.xpos[e.base][2]),heat=e.heat_inst.tolist(),action=action.tolist()))
     if fell:break
    record=dict(seed=seed,seconds=rows[-1]['t'],fell=bool(fell),gate=stand_gate(rows),original_start_tested=False,catalogue_pass=False,rows=rows);log.write(json.dumps(record)+'\n');log.flush();print({k:v for k,v in record.items() if k!='rows'},flush=True)
 finally:plant.close()
