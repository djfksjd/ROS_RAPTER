from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
from contact_wrench import contact_wrench
from stand_observer import StandObserverController
from motion_transition import stand_gate
source=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009')
candidate=next(x for x in map(json.loads,Path('docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1)
design=np.load('docs/evidence/153-stand-observer/observer_design.npz')
paths=[('observer_reset_at_entry',Path('sim/rl/runs/screen_logs/stand_shooting_peak_powell_20261009/initial_parameters.npy'))]
K=np.load('sim/rl/runs/screen_logs/local_equilibrium_linearization/lqr_design.npz')['K']
with Path('sim/rl/runs/screen_logs/stand_blended_handoff_pair.diagnostic.jsonl').open('x') as log:
 for cap_time in [1.6]:
  for name,path in paths:
   parameters=np.load(path).reshape(4,5)
   for seed in [83002,86013]:
    plant=LocalStandModel(source,candidate);plant.set_stand_current_limit(.995 if cap_time==0. else None);e=plant.env;m=e.model;e.init_seed=seed;e.reset();e.command=np.zeros(3);e.resample_steps=0;obs=e._obs();controller=StandObserverController(design);rows=[];entered=False;entry_time=None
    lower=np.maximum(-1.,(e.lo[e.policy_idx]-e.q0[e.policy_idx])/e.scale[e.policy_idx]);upper=np.minimum(1.,(e.hi[e.policy_idx]-e.q0[e.policy_idx])/e.scale[e.policy_idx])
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
       if name=='observer_reset_at_entry':controller.reset()
      proposed=controller.act(obs)
      if entered:
       action=np.clip(plant.u-K@plant.state() if name=='full_state_at_entry' else proposed,lower,upper)
      else:
       common,spread,hp,knee,tail=[np.interp(step*.02,[0.,.2,.4,.6],parameters[:,j]) for j in range(5)]
       action=np.clip([common+spread,hp,knee,common-spread,hp,knee,0.,tail],lower,upper)
      if entered:
       blend=min(max((step*.02-.6)/.2,0.),1.)
       action=np.clip(plant.u+blend*(action-plant.u),lower,upper)
      controller.prior+=controller.B@(action-proposed)
      before=e.data.xpos[e.base][:2].copy();obs,_,fell,_,info=e.step(action)
      rows.append(dict(t=(step+1)*.02,before_position_xy=before.tolist(),position_xy=e.data.xpos[e.base][:2].tolist(),tilt=float(info['tilt']),flight=bool(info['flight']),speed_xy=float(np.linalg.norm(info['v_body'][:2])),heat=e.heat_inst.tolist(),height=float(e.data.xpos[e.base][2]),action=action.tolist(),pre_foot_load_N=loads.tolist(),entered=entered))
      if fell:break
     load=float(np.max(np.mean([r['heat'] for r in rows[-2000:]],axis=0)[e.motor_cont<1e8])) if len(rows)>=3000 else None;gate=stand_gate(rows)
     r=dict(candidate=name,trajectory=path.name,continuous_cap_start_seconds=cap_time,seed=seed,seconds=rows[-1]['t'],fell=bool(fell),entry_time=entry_time,gate=gate,max_load_last40=load,original_start_stand_pass=bool(len(rows)==3000 and not fell and gate['pass_gate'] and load<=1.),catalogue_pass=False,contact_gate_uses_simulator_force=False,fixed_time_handoff_diagnostic=True,full_state_control_after_entry=(name=='full_state_at_entry'),rows=rows)
     log.write(json.dumps(r,allow_nan=False)+'\n');log.flush();print({k:v for k,v in r.items() if k!='rows'},flush=True)
    finally:plant.close()
