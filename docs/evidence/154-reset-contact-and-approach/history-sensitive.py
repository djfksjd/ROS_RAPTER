from pathlib import Path
import sys,json,hashlib
import numpy as np
import mujoco
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
from stand_observer import StandObserverController
from contact_wrench import contact_wrench
source=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009')
candidate=next(x for x in map(json.loads,Path('docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1)
design=np.load('docs/evidence/153-stand-observer/observer_design.npz')
plant=LocalStandModel(source,candidate);plant.set_stand_current_limit(.995);e=plant.env;m=e.model
original_step=mujoco.mj_step

def residual(d):
 result={}
 for side in ['left','right']:
  ai=e.ix[side+'_ankle_pitch_joint'];ki=e.ix[side+'_knee_pitch_joint'];ri=e.ix[side+'_ankle_roll_joint']
  result[side+'_pitch']=float(d.qpos[e.q_adr[ai]]+d.qpos[e.q_adr[ki]]-e.q0[ai]-e.q0[ki])
  result[side+'_roll']=float(d.qpos[e.q_adr[ri]])
 return result

def sample(d):
 mask=d.efc_type[:d.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY
 force,moment,contacts=contact_wrench(m,d,e.root)
 return dict(time=float(d.time),equality_residual=residual(d),equality_force=d.efc_force[:d.nefc][mask].tolist(),max_abs_qacc=float(np.max(np.abs(d.qacc))),knee_passive_torque=[float(d.qfrc_passive[e.v_adr[e.ix[s+'_knee_pitch_joint']]]) for s in ['left','right']],force_world=force.tolist(),qvel=d.qvel.tolist())
try:
 with Path('sim/rl/runs/screen_logs/reset_constraint_probe.diagnostic.jsonl').open('x') as f:
  for mode,telemetry,seeds in [('original',False,[83001]),('original',True,range(83001,83005)),('dependent_projected',True,range(83001,83005))]:
   for seed in seeds:
    e.init_seed=seed;e.reset();e.command=np.zeros(3);e.resample_steps=0
    q_before=e.data.qpos.copy();v_before=e.data.qvel.copy()
    if mode=='dependent_projected':
     for side in ['left','right']:
      ai=e.ix[side+'_ankle_pitch_joint'];ki=e.ix[side+'_knee_pitch_joint'];ri=e.ix[side+'_ankle_roll_joint']
      e.data.qpos[e.q_adr[ai]]=e.q0[ai]+e.q0[ki]-e.data.qpos[e.q_adr[ki]]
      e.data.qpos[e.q_adr[ri]]=0.
     mujoco.mj_forward(m,e.data)
    e._clamp()
    audit=mujoco.MjData(m);audit.qpos[:]=e.data.qpos;audit.qvel[:]=e.data.qvel;audit.ctrl[:]=e.data.ctrl;mujoco.mj_forward(m,audit)
    initial=sample(audit);initial['qpos_change']=(e.data.qpos-q_before).tolist();initial['qvel_unchanged']=bool(np.array_equal(v_before,e.data.qvel));initial['initial_vx']=float(e.data.qvel[0])
    controller=StandObserverController(design);obs=e._obs();rows=[];sub=[]
    def instrument(mm,dd):
     original_step(mm,dd)
     if dd.time<=.200001:sub.append(sample(dd))
    mujoco.mj_step=instrument if telemetry else original_step
    for step in range(100):
     action=controller.act(obs);obs,_,fell,_,info=e.step(action)
     rows.append(dict(t=(step+1)*.02,qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist(),action=action.tolist(),tilt=float(info['tilt']),flight=bool(info['flight'])))
     if fell:break
    record=dict(mode=mode,telemetry=telemetry,seed=seed,initial=initial,seconds=rows[-1]['t'],fell=bool(fell),rows=rows,first200ms=sub,catalogue_pass=False)
    f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
    print({k:v for k,v in record.items() if k not in ['initial','rows','first200ms']},flush=True)
finally:
 mujoco.mj_step=original_step;plant.close()
