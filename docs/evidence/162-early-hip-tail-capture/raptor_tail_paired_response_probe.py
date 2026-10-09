from pathlib import Path
import sys,json,numpy as np,mujoco
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
candidate=next(x for x in map(json.loads,Path('docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1)
p=LocalStandModel('sim/rl/runs/kl005_lr3e5_update64k_20261009',candidate);results=[]
source=[json.loads(x) for x in Path('docs/evidence/158-contact-and-handoff/contact-pair.jsonl').read_text().splitlines()]
try:
 for r in source:
  if not r['telemetry']:continue
  for t in [.02,.1]:
   snap=min(r['substeps'],key=lambda s:abs(s['state_time']-t));dq=np.zeros(p.nv);mujoco.mj_differentiatePos(p.env.model,dq,1.,p.q,np.array(snap['qpos_after']))
   previous=np.array(r['rows'][round(t/.02)-1]['action']);x=np.r_[dq,snap['qvel_after'],previous-p.u];u=np.array(r['rows'][round((t+.02)/.02)-1]['action'])
   for mode in ['baseline','tail_plus','tail_minus','hip_plus','hip_minus','paired_brake']:
    action=u.copy()
    if mode.startswith('tail'):action[7]+=.025 if mode.endswith('plus') else -.025
    if mode.startswith('hip'):action[[1,4]]+=.025 if mode.endswith('plus') else -.025
    if mode=='paired_brake':
     ratio=sum(p.env.kp[p.env.policy_idx[i]]*p.env.scale[p.env.policy_idx[i]] for i in [1,4])/(p.env.kp[p.env.policy_idx[7]]*p.env.scale[p.env.policy_idx[7]])
     action[[1,4]]-=.025;action[7]+=.025*ratio
    flag=None
    try:p.transition(x,action-p.u)
    except ValueError as error:flag=str(error)
    out=p.state();dd=p.env.data;R=dd.xmat[p.env.base].reshape(3,3);offset=p.env.model.body_pos[p.env.base];body_velocity=R.T@dd.qvel[:3]+np.cross(dd.qvel[3:6],offset)
    results.append(dict(seed=r['seed'],time=t,mode=mode,root_pitch_rate=float(out[p.nv+4]),root_forward_velocity=float(out[p.nv]),body_point_forward_velocity=float(body_velocity[0]),region_error=flag,offline_true_state_initialized=True))
finally:p.close()
Path('sim/rl/runs/screen_logs/tail_paired_response_probe.json').write_text(json.dumps(results,indent=2)+'\n')
for seed in [83002,86013]:
 for t in [.02,.1]:
  rows=[r for r in results if r['seed']==seed and r['time']==t];baseline=next(r for r in rows if r['mode']=='baseline')
  print(seed,t,[(r['mode'],round(r['root_pitch_rate']-baseline['root_pitch_rate'],5),round(r['body_point_forward_velocity']-baseline['body_point_forward_velocity'],5),r['region_error']) for r in rows],flush=True)
