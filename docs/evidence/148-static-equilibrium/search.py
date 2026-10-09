from pathlib import Path
import sys,json
import numpy as np
import mujoco
from scipy.optimize import least_squares
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv,LINK_ETA
from train_run import env_kwargs
from contact_wrench import contact_wrench
source=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009')
e=RunEnv(**env_kwargs(json.loads((source/'args.json').read_text())),randomize=False,seed=7);e.reset();e.command=np.zeros(3)
m=e.model;d=mujoco.MjData(m);template=e.data.qpos.copy();pi=e.policy_idx;actdof=e.v_adr[pi];unact=np.array([i for i in range(m.nv) if i not in actdof])
qaddress=lambda n:int(m.jnt_qposadr[m.joint(n).id])
toes=[('left_toe_2','right_toe_2'),('left_toe_1','right_toe_3'),('left_toe_3','right_toe_1')]
lo=np.array([-1.5273,.5,-.04,-.3,-.03,-np.radians(10),-.15,-.6,-.6,-.6,-.6,-.3,-.3])
hi=np.array([.4727,2.5,.04,.3,.03,np.radians(10),.45,.9,.9,.9,.9,.3,.3])
def setup(x):
 d.qpos[:]=template;d.qvel[:]=0.;d.qacc[:]=0.;d.qfrc_applied[:]=0.;d.xfrc_applied[:]=0.;d.qacc_warmstart[:]=0.
 hp,kn,ad,hr,ar,pitch,z=x[:7];d.qpos[:3]=[-.58*np.sin(pitch),0.,z];d.qpos[3:7]=[np.cos(pitch/2),0.,np.sin(pitch/2),0.]
 for side,sgn in [('left',1.),('right',-1.)]:
  d.qpos[qaddress(side+'_hip_pitch_joint')]=hp;d.qpos[qaddress(side+'_knee_pitch_joint')]=kn
  i=e.ix[side+'_ankle_pitch_joint'];ki=e.ix[side+'_knee_pitch_joint'];d.qpos[qaddress(side+'_ankle_pitch_joint')]=e.q0[i]+e.q0[ki]-kn+ad
  d.qpos[qaddress(side+'_hip_roll_joint')]=sgn*hr;d.qpos[qaddress(side+'_ankle_roll_joint')]=sgn*ar
 for t,(left,right) in enumerate(toes):
  for k,part in enumerate(['proximal','distal']):
   for n in [left,right]:d.qpos[qaddress(n+'_'+part+'_joint')]=x[7+2*t+k]
 mujoco.mj_inverse(m,d)
 tau=d.qfrc_inverse[actdof].copy();g=e.gear_const.copy();g[e.fourbar_idx]=np.interp(d.qpos[e.q_adr[e.fourbar_idx]],*e.knee_ratio)
 cont=(g*LINK_ETA*e.motor_cont)[pi];target=d.qpos[e.q_adr[pi]]+tau/e.kp[pi];action=(target-e.q0[pi])/e.scale[pi]
 force,moment,contacts=contact_wrench(m,d,e.root);feet={s:0. for s in ['left','right']};bad=0.
 for c in contacts:
  n=m.body(c['body']).name;isfoot=any(k in n for k in ['foot','toe','metatarsus'])
  if not isfoot:bad+=abs(c['force_world_N'][2])
  else:
   for s in feet:
    if n.startswith(s):feet[s]+=max(0.,c['force_world_N'][2])
 return tau,cont,action,feet,bad

def residual(x):
 tau,cont,action,feet,bad=setup(x)
 scale=np.full(m.nv,10.);scale[:3]=e.weight;scale[3:6]=30.
 return np.r_[10*d.qfrc_inverse[unact]/scale[unact],2*np.maximum(np.abs(tau/cont)-1.,0.),2*np.maximum(np.abs(action)-1.,0.),5*bad/e.weight,[max(0.,.15-feet[s]/e.weight) for s in feet],.02*(d.xpos[e.base][2]-.744),.001*(x-template[0])]
p=Path('sim/rl/runs/screen_logs/static_equilibrium_search.diagnostic.jsonl')
try:
 with p.open('x') as f:
  for knee in [1.5,1.25,1.1]:
   x=np.zeros(13);x[0]=-.5273;x[1]=knee;x[6]=template[2]-.01
   result=least_squares(residual,x,bounds=(lo,hi),max_nfev=400,ftol=1e-10,xtol=1e-10,gtol=1e-10)
   tau,cont,action,feet,bad=setup(result.x);q=d.qpos.copy();inverse=d.qfrc_inverse.copy()
   record=dict(start_knee=knee,status=int(result.status),message=result.message,nfev=result.nfev,cost=float(result.cost),parameters=result.x.tolist(),qpos=q.tolist(),unactuated_dofs=unact.tolist(),inverse_force=inverse.tolist(),actuated_dofs=actdof.tolist(),joint_names=[e.active[i] for i in pi],required_joint_torque=tau.tolist(),continuous_joint_torque=cont.tolist(),action=action.tolist(),foot_vertical_forces=feet,nonfoot_vertical_force=bad,torso_height=float(d.xpos[e.base][2]),static_feasible=bool(np.max(np.abs(inverse[:3]))<1 and np.max(np.abs(inverse[3:6]))<.1 and np.max(np.abs(inverse[unact[unact>=6]]))<.1 and np.max(np.abs(tau/cont))<=1 and np.max(np.abs(action))<=1 and min(feet.values())>=.15*e.weight and bad<1e-3),forward_validation_performed=False)
   f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();print({k:v for k,v in record.items() if k not in ['qpos','inverse_force','parameters','unactuated_dofs','actuated_dofs']},flush=True)
finally:e.close()
