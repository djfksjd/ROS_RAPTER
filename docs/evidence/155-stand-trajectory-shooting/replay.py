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
paths=[('relative_best',Path('sim/rl/runs/screen_logs/stand_shooting_20261009/best_parameters.npy')),('absolute_initial',Path('sim/rl/runs/screen_logs/stand_shooting_absolute_20261009/initial_parameters.npy')),('absolute_best',Path('sim/rl/runs/screen_logs/stand_shooting_absolute_20261009/best_parameters.npy')),('absolute_final',Path('sim/rl/runs/screen_logs/stand_shooting_absolute_20261009/final_parameters.npy'))]
with Path('sim/rl/runs/screen_logs/stand_shooting_replay.diagnostic.jsonl').open('x') as log:
 for name,path in paths:
  parameters=np.load(path).reshape(4,5)
  for seed in range(83001,83005):
   plant=LocalStandModel(source,candidate);plant.set_stand_current_limit(.995);e=plant.env;m=e.model;e.init_seed=seed;e.reset();e.command=np.zeros(3);e.resample_steps=0;obs=e._obs();controller=StandObserverController(design);rows=[];entered=False;entry_time=None
   lower=np.maximum(-1.,(e.lo[e.policy_idx]-e.q0[e.policy_idx])/e.scale[e.policy_idx]);upper=np.minimum(1.,(e.hi[e.policy_idx]-e.q0[e.policy_idx])/e.scale[e.policy_idx])
   try:
    for step in range(3000):
     _,_,contacts=contact_wrench(m,e.data,e.root);loads=np.zeros(2)
     for c in contacts:
      body=m.body(c['body']).name
      if any(x in body for x in ['foot','toe','metatarsus']):
       for i,side in enumerate(['left','right']):
        if body.startswith(side):loads[i]+=max(0.,c['force_world_N'][2])
     ready=bool(step>=30 and min(loads)>=.15*e.weight and np.max(np.abs(obs[12:24]-controller.reference[12:24]))<=.08 and np.max(np.abs(obs[24:36]))/.05<=2. and np.linalg.norm(obs[3:6]-controller.reference[3:6])<=.1 and np.linalg.norm(obs[9:12])/.1<=.5)
     if not entered and ready:entered=True;entry_time=step*.02
     proposed=controller.act(obs)
     if entered:action=np.clip(proposed,lower,upper)
     else:
      common,spread,hp,knee,tail=[np.interp(step*.02,[0.,.2,.4,.6],parameters[:,j]) for j in range(5)]
      action=np.clip([common+spread,hp,knee,common-spread,hp,knee,0.,tail],lower,upper)
     controller.prior+=controller.B@(action-proposed)
     before=e.data.xpos[e.base][:2].copy();obs,_,fell,_,info=e.step(action)
     rows.append(dict(t=(step+1)*.02,before_position_xy=before.tolist(),position_xy=e.data.xpos[e.base][:2].tolist(),tilt=float(info['tilt']),flight=bool(info['flight']),speed_xy=float(np.linalg.norm(info['v_body'][:2])),heat=e.heat_inst.tolist(),height=float(e.data.xpos[e.base][2]),action=action.tolist(),pre_foot_load_N=loads.tolist(),entered=entered))
     if fell:break
    load=float(np.max(np.mean([r['heat'] for r in rows[-2000:]],axis=0)[e.motor_cont<1e8])) if len(rows)>=3000 else None;gate=stand_gate(rows)
    r=dict(candidate=name,seed=seed,seconds=rows[-1]['t'],fell=bool(fell),entry_time=entry_time,gate=gate,max_load_last40=load,original_start_stand_pass=bool(len(rows)==3000 and not fell and gate['pass_gate'] and load<=1.),catalogue_pass=False,contact_gate_uses_simulator_force=True,rows=rows)
    log.write(json.dumps(r,allow_nan=False)+'\n');log.flush();print({k:v for k,v in r.items() if k!='rows'},flush=True)
   finally:plant.close()
