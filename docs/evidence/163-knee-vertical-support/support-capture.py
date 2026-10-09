from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
from contact_wrench import contact_wrench
from stand_observer import StandObserverController
from motion_transition import stand_gate
from servo_target_limit import limit_servo_action
from run_env import motor_available,MOTOR_CLASSES,LINK_ETA,joint_type
source=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009')
candidate=next(x for x in map(json.loads,Path('docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1)
design=np.load('docs/evidence/153-stand-observer/observer_design.npz')
paths=[(mode,Path('sim/rl/runs/screen_logs/stand_shooting_peak_powell_20261009/initial_parameters.npy')) for mode in ['support_only','paired_pitch_support']]
K=np.load('sim/rl/runs/screen_logs/local_equilibrium_linearization/lqr_design.npz')['K']
with Path('sim/rl/runs/screen_logs/stand_knee_support_capture.diagnostic.jsonl').open('x') as log:
 for cap_time in [1.6]:
  for name,path in paths:
   parameters=np.load(path).reshape(4,5)
   for seed in [83002,*range(86001,86021)]:
    plant=LocalStandModel(source,candidate);plant.set_stand_current_limit(.995 if cap_time==0. else None);e=plant.env;m=e.model;e.init_seed=seed;e.reset();e.command=np.zeros(3);e.resample_steps=0;obs=e._obs();controller=StandObserverController(design);rows=[];entered=False;entry_time=None
    pi=e.policy_idx;lower=np.maximum(-1.,(e.lo[e.policy_idx]-e.q0[e.policy_idx])/e.scale[e.policy_idx]);upper=np.minimum(1.,(e.hi[e.policy_idx]-e.q0[e.policy_idx])/e.scale[e.policy_idx])
    try:
     for step in range(3000):
      if cap_time is not None and cap_time>0. and step==round(cap_time/.02):plant.set_stand_current_limit(.995)
      _,_,contacts=contact_wrench(m,e.data,e.root);loads=np.zeros(2)
      for c in contacts:
       body=m.body(c['body']).name
       if any(x in body for x in ['foot','toe','metatarsus']):
        for i,side in enumerate(['left','right']):
         if body.startswith(side):loads[i]+=max(0.,c['force_world_N'][2])
      ready=bool(step>=30) # Reproduce the offline planner fixed-time handoff; not an operating gate.
      if not entered and ready:
       entered=True;entry_time=step*.02
       controller.reset()
      proposed=controller.act(obs)
      if entered:
       action=np.clip(plant.u-K@plant.state() if name=='full_state_at_entry' else proposed,lower,upper)
      else:
       common,spread,hp,knee,tail=[np.interp(step*.02,[0.,.2,.4,.6],parameters[:,j]) for j in range(5)]
       action=np.clip([common+spread,hp,knee,common-spread,hp,knee,0.,tail],lower,upper)
      infeasible=np.zeros(8,dtype=bool)
      if True:
       observed_q=e.q0+obs[12:24];observed_qd=obs[24:36]/.05;tl=[];th=[]
       for j in e.policy_idx:
        cls,g=e.arch8_map[joint_type(e.active[j])]
        gear=float(np.interp(observed_q[j],*e.knee_ratio)) if g=='fourbar' else g
        drive=gear*LINK_ETA*float(motor_available(cls,gear*observed_qd[j]));brake=gear*LINK_ETA*MOTOR_CLASSES[cls][0]
        low,high=(-brake,drive) if observed_qd[j]>=0 else (-drive,brake)
        if step>=80:
         cap=.995*gear*LINK_ETA*e.motor_cont[j];low=max(low,-cap);high=min(high,cap)
        tl.append(low);th.append(high)
       if entered:action,infeasible=limit_servo_action(action,q=observed_q[pi],qd=observed_qd[pi],q0=e.q0[pi],scale=e.scale[pi],kp=e.kp[pi],kd=e.kd[pi],torque_lower=tl,torque_upper=th,action_lower=lower,action_upper=upper)
      hip_delta=0.;body_vx=None;tail_base_tau=None
      if not entered and name.startswith('paired'):
       pitch=np.arcsin(np.clip(obs[3],-1.,1.));pitch_ref=np.arcsin(np.clip(controller.reference[3],-1.,1.))
       roll=np.arctan2(-obs[4],-obs[5]);roll_ref=np.arctan2(-controller.reference[4],-controller.reference[5]);omega=obs[:3]/.25
       roll_delta=np.clip(.2*(roll-roll_ref)+.04*omega[0],-.15,.15)
       action[7]+=np.clip(.25*(pitch-pitch_ref)+.05*omega[1],-.25,.25)/e.scale[pi[7]]
       j=pi[7];gain=e.kp[j]*e.scale[j]
       lo=max(lower[7],(observed_q[j]+(tl[7]+e.kd[j]*observed_qd[j])/e.kp[j]-e.q0[j])/e.scale[j])
       hi=min(upper[7],(observed_q[j]+(th[7]+e.kd[j]*observed_qd[j])/e.kp[j]-e.q0[j])/e.scale[j])
       assert lo<=hi, 'empty early tail action interval'
       action[7]=np.clip(action[7],lo,hi)
       tail_base_tau=e.kp[j]*(e.q0[j]+e.scale[j]*action[7]-observed_q[j])-e.kd[j]*observed_qd[j]
       body_vx=float((obs[9:12]/.1+np.cross(omega,m.body_pos[e.base]))[0])
       if name.startswith('paired'):
        hip_gain=sum(e.kp[pi[i]]*e.scale[pi[i]] for i in [1,4]);ratio=hip_gain/gain
        low_delta=max(-.025,(tail_base_tau-th[7])/hip_gain,*[lower[i]-action[i] for i in [1,4]])
        high_delta=min(.025,(tail_base_tau-tl[7])/hip_gain,*[upper[i]-action[i] for i in [1,4]])
        assert low_delta<=high_delta, 'empty paired allocation'
        hip_delta=float(np.clip(-(.02/(.15*1.6))*body_vx,low_delta,high_delta))
        action[[1,4]]+=hip_delta;action[7]-=ratio*hip_delta
       action=np.clip(action,lower,upper)
      knee_delta=0.
      if not entered:
       knee_delta=-.01*float(np.interp(step*.02,[0.,.18,.22,.30,.34,.6],[0.,0.,1.,1.,0.,0.]))
       action[[2,5]]+=knee_delta;action=np.clip(action,lower,upper)
      controller.prior+=controller.B@(action-proposed)
      before=e.data.xpos[e.base][:2].copy();obs,_,fell,_,info=e.step(action)
      rows.append(dict(t=(step+1)*.02,before_position_xy=before.tolist(),position_xy=e.data.xpos[e.base][:2].tolist(),tilt=float(info['tilt']),flight=bool(info['flight']),speed_xy=float(np.linalg.norm(info['v_body'][:2])),heat=e.heat_inst.tolist(),height=float(e.data.xpos[e.base][2]),action=action.tolist(),pre_foot_load_N=loads.tolist(),entered=entered,early_body_point_vx=body_vx,early_hip_delta=hip_delta,knee_support_delta=knee_delta,early_tail_base_tau=tail_base_tau,infeasible_target=infeasible.tolist()))
      if fell:break
     load=float(np.max(np.mean([r['heat'] for r in rows[-2000:]],axis=0)[e.motor_cont<1e8])) if len(rows)>=3000 else None;gate=stand_gate(rows)
     r=dict(candidate=name,trajectory=path.name,continuous_cap_start_seconds=cap_time,seed=seed,seconds=rows[-1]['t'],fell=bool(fell),entry_time=entry_time,gate=gate,prefix_gate=stand_gate(rows[:30]),max_load_last40=load,original_start_stand_pass=bool(len(rows)==3000 and not fell and gate['pass_gate'] and load<=1.),catalogue_pass=False,contact_gate_uses_simulator_force=False,fixed_time_handoff_diagnostic=True,full_state_control_after_entry=(name=='full_state_at_entry'),rows=rows)
     log.write(json.dumps(r,allow_nan=False)+'\n');log.flush();print({k:v for k,v in r.items() if k!='rows'},flush=True)
    finally:plant.close()
