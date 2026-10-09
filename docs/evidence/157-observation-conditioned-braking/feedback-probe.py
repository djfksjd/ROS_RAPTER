from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
from stand_observer import StandObserverController
from braking_approach import BrakingApproach
from motion_transition import stand_gate
source=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009')
candidate=next(x for x in map(json.loads,Path('docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1)
design=np.load('docs/evidence/153-stand-observer/observer_design.npz')
parameters=np.load('docs/evidence/156-stand-peak-envelope/search/initial_parameters.npy').reshape(4,5)
with Path('sim/rl/runs/screen_logs/braking_feedback_probe.diagnostic.jsonl').open('x') as log:
 for mode in ['pose_feedback','pose_velocity_feedback']:
  for seed in [83002,*range(86001,86021)]:
   plant=LocalStandModel(source,candidate);e=plant.env;m=e.model;e.init_seed=seed;e.reset();e.command=np.zeros(3);e.resample_steps=0;obs=e._obs();controller=StandObserverController(design)
   pi=e.policy_idx;lower=np.maximum(-1.,(e.lo[pi]-e.q0[pi])/e.scale[pi]);upper=np.minimum(1.,(e.hi[pi]-e.q0[pi])/e.scale[pi])
   stiffness=np.zeros(8);rest=np.zeros(8)
   for i,j in enumerate(pi):
    name=e.active[j]
    if 'knee_pitch' in name:
     t=m.tendon('spring_'+name.removesuffix('_joint')).id;stiffness[i]=e.spring_engage[(name.split('_')[0],'knee_pitch')][1];rest[i]=m.tendon_lengthspring[t][0]
   approach=BrakingApproach(obs,q0=e.q0,kp=e.kp,scale=e.scale,policy_indices=pi,spring_stiffness=stiffness,spring_rest=rest,reference_action=plant.u,lower=lower,upper=upper,symmetric_pairs=[(1,4),(2,5)] if mode=='observed_symmetric' else [])
   rows=[]
   try:
    for step in range(3000):
     if step==80:plant.set_stand_current_limit(.995)
     if step==30:controller.reset()
     proposed=controller.act(obs)
     if step>=30:action=np.clip(proposed,lower,upper)
     else:
      common,spread,hp,knee,tail=[np.interp(step*.02,[0.,.2,.4,.6],parameters[:,j]) for j in range(5)];action=np.clip([common+spread,hp,knee,common-spread,hp,knee,0.,tail],lower,upper)
     if step<30:
      pitch=np.arcsin(np.clip(obs[3],-1.,1.));pitch_ref=np.arcsin(np.clip(controller.reference[3],-1.,1.))
      roll=np.arctan2(-obs[4],-obs[5]);roll_ref=np.arctan2(-controller.reference[4],-controller.reference[5])
      omega=obs[:3]/.25
      pitch_delta=.25*(pitch-pitch_ref)+.05*omega[1]
      if mode=='pose_velocity_feedback':pitch_delta-=.1/.744*(obs[9]/.1)
      roll_delta=np.clip(.2*(roll-roll_ref)+.04*omega[0],-.15,.15)
      pitch_delta=np.clip(pitch_delta,-.25,.25)
      action=action.copy()
      for i,j in enumerate(pi):
       if 'hip_pitch' in e.active[j]:action[i]+=pitch_delta/e.scale[j]
       if 'hip_roll' in e.active[j]:action[i]+=roll_delta/e.scale[j]
      action=np.clip(action,lower,upper)
     controller.prior+=controller.B@(action-proposed)
     before=e.data.xpos[e.base][:2].copy();obs,_,fell,_,info=e.step(action)
     rows.append(dict(t=(step+1)*.02,before_position_xy=before.tolist(),position_xy=e.data.xpos[e.base][:2].tolist(),tilt=float(info['tilt']),flight=bool(info['flight']),speed_xy=float(np.linalg.norm(info['v_body'][:2])),heat=e.heat_inst.tolist(),height=float(e.data.xpos[e.base][2]),action=action.tolist()))
     if fell:break
    load=float(np.max(np.mean([r['heat'] for r in rows[-2000:]],axis=0)[e.motor_cont<1e8])) if len(rows)==3000 else None;gate=stand_gate(rows)
    r=dict(mode=mode,seed=seed,seconds=rows[-1]['t'],fell=bool(fell),gate=gate,max_load_last40=load,pass_stand=bool(len(rows)==3000 and not fell and gate['pass_gate'] and load<=1.),initial_action=approach.initial.tolist(),catalogue_pass=False,rows=rows)
    log.write(json.dumps(r,allow_nan=False)+'\n');log.flush();print({k:v for k,v in r.items() if k not in ['rows','initial_action']},flush=True)
   finally:plant.close()
