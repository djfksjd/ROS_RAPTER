from pathlib import Path
import sys,json
import numpy as np
import mujoco
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load
from contact_wrench import contact_wrench
from actuator_turn_probe import utilization
run=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009')
class Adapter:
 def __init__(self,model):self.model=model
 def predict(self,obs,**kw):
  x=np.array(obs,copy=True);assert x.shape[1]==46 and np.isfinite(x).all();mask=x[:,8]!=0;assert np.all(x[mask,6]==np.float32(.4))
  x[mask,8]*=np.float32(1.1)
  x[mask,6]+=np.asarray(.1*np.clip(.8+1.2*(4.-x[mask,9]*10.),0.,1.5),dtype=np.float32)
  return self.model.predict(x,**kw)
def probe(yaw,enabled):
 kw=env_kwargs(json.loads((run/'args.json').read_text()));env=RunEnv(**{**kw,'episode_s':15.,'level':0.},randomize=False,seed=7);env.init_seed=71001
 model,norm=load(str(run/'model.zip'),env);model=Adapter(model);norm.reset();env.resample_steps=0
 original=mujoco.mj_step;scratch=mujoco.MjData(env.model);physics=[];controls=[]
 ids=env.act[env.policy_idx];mass=float(env.model.body_subtreemass[env.root])
 def state(m,d):
  mujoco.mj_copyData(scratch,m,d);mujoco.mj_kinematics(m,scratch);mujoco.mj_comPos(m,scratch);mujoco.mj_comVel(m,scratch);mujoco.mj_subtreeVel(m,scratch)
  e=scratch.xmat[env.base].reshape(3,3)[:,0].copy();e[2]=0.;e/=np.linalg.norm(e)
  return mass*scratch.subtree_linvel[env.root].copy(),e
 def measured(m,d):
  time=round(float(d.time),12);active=4.<=time<6.
  if active:
   before,e=state(m,d);low,high=m.actuator_forcerange[ids].T.copy()
  original(m,d)
  if active:
   force,moment,contacts=contact_wrench(m,d,env.root);after,ea=state(m,d);dt=m.opt.timestep
   external=force+mass*m.opt.gravity;dp=after-before
   eperp=np.array([-e[1],e[0],0.]);vmid=(before+after)/(2*mass)
   physics.append(dict(contacts=[{**c,'body_name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,c['body'])} for c in contacts],longitudinal_com_work_J=float(force@e)*float(vmid@e)*dt,lateral_com_work_J=float(force@eperp)*float(vmid@eperp)*dt,delta_com_horizontal_kinetic_J=float((after[:2]@after[:2]-before[:2]@before[:2])/(2*mass)),com_speed_before_mps=float(np.linalg.norm(before[:2])/mass),com_speed_after_mps=float(np.linalg.norm(after[:2])/mass),com_forward_after_mps=float(after@ea/mass),com_lateral_after_mps=float(after@np.array([-ea[1],ea[0],0.])/mass),t=time,force_world_N=force.tolist(),longitudinal_force_N=float(force@e),lateral_force_N=float(force@np.array([-e[1],e[0],0.])),delta_momentum_world_Ns=dp.tolist(),residual_world_Ns=(dp-external*dt).tolist(),delta_forward_momentum_Ns=float(after@ea-before@e),rotation_projection_Ns=float(after@(ea-e)),torque_utilization=utilization(d.actuator_force[ids],low,high).tolist()))
 try:
  if enabled:mujoco.mj_step=measured
  for i in range(500):
   time=round(float(env.data.time),12);env.command=np.array([4.,0.,yaw if 4.<=time<6. else 0.]);a,_=model.predict(norm.normalize_obs(env._obs()[None]),deterministic=True)
   _,_,fell,_,info=env.step(a[0]);controls.append(dict(t=time,vx=float(info['v_body'][0]),vy=float(info['v_body'][1]),qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist(),action=a[0].tolist()))
   if fell:break
  summary={}
  if physics:
   dt=env.model.opt.timestep;u=np.array([r['torque_utilization'] for r in physics]);res=np.sum([r['residual_world_Ns'] for r in physics],axis=0)
   summary=dict(contact_longitudinal_impulse_Ns=sum(r['longitudinal_force_N']*dt for r in physics),contact_lateral_impulse_Ns=sum(r['lateral_force_N']*dt for r in physics),forward_momentum_delta_Ns=sum(r['delta_forward_momentum_Ns'] for r in physics),rotation_projection_Ns=sum(r['rotation_projection_Ns'] for r in physics),total_world_balance_residual_Ns=res.tolist(),joints=[dict(name=env.active[j],near_bound_fraction=float(np.mean(u[:,i]>=.98)),max_utilization=float(u[:,i].max())) for i,j in enumerate(env.policy_idx)])
  return dict(yaw=yaw,enabled=enabled,seed=71001,mass=mass,fell=bool(fell),summary=summary,controls=controls,physics=physics)
 finally:mujoco.mj_step=original;norm.close()
out=Path('sim/rl/runs/screen_logs/low_lr_combined_contact.diagnostic.jsonl')
with out.open('x') as f:
 for yaw in [-1.,1.]:
  a=probe(yaw,False);b=probe(yaw,True);assert a['controls']==b['controls'],'probe changed physics';b['exact_control_parity']=True
  f.write(json.dumps(b,allow_nan=False)+'\n');f.flush();print({k:v for k,v in b.items() if k not in ['physics','controls']},flush=True)
