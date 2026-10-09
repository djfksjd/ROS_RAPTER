from pathlib import Path
import json,sys,numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
c=next(x for x in map(json.loads,Path('docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1)
p=LocalStandModel('sim/rl/runs/kl005_lr3e5_update64k_20261009',c);e=p.env
rows=[json.loads(x) for x in Path('sim/rl/runs/screen_logs/stand_contact_pair_probe.diagnostic.jsonl').read_text().splitlines()]
summary={'telemetry_exact':True,'force_timestamp':'before physics integration, state_time minus1ms','state_timestamp':'after physics integration','pair':[]}
try:
 for seed in [83002,86013]:
  off,on=[x for x in rows if x['seed']==seed];assert off['rows']==on['rows']
  sub=on['substeps'];window=[x for x in sub if x['force_time']<.2];load=np.array([x['foot_load_N'] for x in window]);mom=np.array([x['moment_about_com_Nm'] for x in window])
  prev=np.array(on['seed_initial_qpos']);stats=[]
  for ss in sub:
   e.data.qpos[:]=prev;e.data.qvel[:]=ss['qvel_before'];lo,hi=e._clamp();tau=np.array(ss['motor_torque']);saturated=(np.isclose(tau,lo,atol=1e-6)|np.isclose(tau,hi,atol=1e-6))[e.policy_idx]
   stats.append((ss['force_time'],saturated));prev=np.array(ss['qpos_after'])
  bands={}
  for a,b in [(0,.02),(.58,.6),(.6,.62),(.62,.64)]:
   selected=[s for t,s in stats if a-1e-8<=t<b-1e-8];bands[f'{a:.2f}-{b:.2f}']=np.mean(selected,axis=0).tolist()
  summary['pair'].append(dict(seed=seed,initial_vx=on['initial_vx'],seconds=on['seconds'],pass_stand=on['original_start_stand_pass'],first15BW_times=[next((x['force_time'] for x in sub if x['foot_load_N'][i]>=.15*e.weight),None) for i in [0,1]],impulse_first200ms_Ns=(.001*load.sum(axis=0)).tolist(),contact_moment_integral_first200ms_Nms=(.001*mom.sum(axis=0)).tolist(),peak_vertical_BW_first200ms=float(max(load.sum(axis=1))/e.weight),motor_names=[e.active[i] for i in e.policy_idx],saturation_fractions=bands,handoff_normalized_action_change=(np.array(on['rows'][30]['action'])-on['rows'][29]['action']).tolist()))
 Path('sim/rl/runs/screen_logs/stand_contact_pair_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
finally:p.close()
print(json.dumps(summary,indent=2))
