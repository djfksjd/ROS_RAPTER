from pathlib import Path
import sys,json
import numpy as np
import mujoco
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs
from contact_wrench import contact_wrench
from actuator_turn_probe import utilization

def hull(points):
 p=sorted(set(tuple(x) for x in points))
 if len(p)<3:return []
 def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
 lo=[];hi=[]
 for x in p:
  while len(lo)>=2 and cross(lo[-2],lo[-1],x)<=0:lo.pop()
  lo.append(x)
 for x in reversed(p):
  while len(hi)>=2 and cross(hi[-2],hi[-1],x)<=0:hi.pop()
  hi.append(x)
 return lo[:-1]+hi[:-1]

def probe(stiffness,seed,enabled):
 cfg=json.loads(Path('/tmp/raptor_stand_knee_'+str(stiffness)+'/args.json').read_text())
 env=RunEnv(**{**env_kwargs(cfg),'episode_s':15.,'level':0.},randomize=False,seed=7);env.init_seed=seed;env.reset();initial_vx=float(env.data.qvel[0]);env.data.qvel[:6]=0.;mujoco.mj_forward(env.model,env.data);env.command=np.zeros(3);env.resample_steps=0
 m,d=env.model,env.data;scratch=mujoco.MjData(m);original=mujoco.mj_step;physics=[];controls=[];ids=env.act[env.policy_idx];jids=env.policy_idx
 def measured(m,d):
  t=float(d.time);mujoco.mj_copyData(scratch,m,d);mujoco.mj_kinematics(m,scratch);mujoco.mj_comPos(m,scratch)
  com=scratch.subtree_com[env.root].copy();q=d.qpos[env.q_adr].copy();qd=d.qvel[env.v_adr].copy();ctrl=d.ctrl[env.act].copy();low,high=m.actuator_forcerange[ids].T.copy();before_tilt=float(np.arccos(np.clip(scratch.xmat[env.base].reshape(3,3)[2,2],-1,1)))
  original(m,d)
  force,moment,contacts=contact_wrench(m,d,env.root);loaded=[c for c in contacts if c['force_world_N'][2]>1.]
  polygon=hull([c['position_world_m'][:2] for c in loaded]);margin=None
  if len(polygon)>=3:
   margins=[]
   for a,b in zip(polygon,polygon[1:]+polygon[:1]):
    a,b=np.array(a),np.array(b);v=b-a;margins.append(float((v[0]*(com[1]-a[1])-v[1]*(com[0]-a[0]))/np.linalg.norm(v)))
   margin=min(margins)
  physics.append(dict(t=t,com_world_m=com.tolist(),force_world_N=force.tolist(),support_contact_centers_xy=polygon,contact_center_margin_m=margin,loaded_contacts=[{**c,'body_name':m.body(c['body']).name} for c in loaded],tilt_rad=before_tilt,q_before=q[jids].tolist(),qd_before=qd[jids].tolist(),target=ctrl[jids].tolist(),motor_torque_Nm=d.actuator_force[ids].tolist(),forcerange_low=low.tolist(),forcerange_high=high.tolist(),near_bound=utilization(d.actuator_force[ids],low,high).tolist(),passive_force_Nm=d.qfrc_passive[env.v_adr[jids]].tolist(),bias_force_Nm=d.qfrc_bias[env.v_adr[jids]].tolist(),constraint_force_Nm=d.qfrc_constraint[env.v_adr[jids]].tolist(),tendon_stiffness=m.tendon_stiffness.tolist()))
 try:
  if enabled:mujoco.mj_step=measured
  fell=False
  for i in range(500):
   _,_,fell,_,info=env.step(np.zeros(len(env.policy_idx)))
   controls.append(dict(t=float(d.time),position_xy=d.xpos[env.base][:2].tolist(),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),action=[0.]*len(env.policy_idx),heat=env.heat_inst.tolist(),tilt=float(info['tilt']),flight=bool(info['flight']),speed_xy=float(np.linalg.norm(info['v_body'][:2]))))
   if fell:break
  return dict(initial_base_speed_overridden=True,original_initial_vx=initial_vx,knee_design_stiffness=stiffness,seed=seed,enabled=enabled,joint_names=[env.active[j] for j in jids],fell=bool(fell),controls=controls,physics=physics)
 finally:mujoco.mj_step=original;env.close()
out=Path('sim/rl/runs/screen_logs/stand_zero_initial_speed.diagnostic.jsonl')
with out.open('x') as f:
 for stiffness in [170,0]:
  for seed in [22001,22002]:
   a=probe(stiffness,seed,False);b=probe(stiffness,seed,True);assert a['controls']==b['controls'];b['exact_control_parity']=True
   f.write(json.dumps(b,allow_nan=False)+'\n');f.flush();print(stiffness,seed,len(b['physics']),b['fell'],flush=True)
