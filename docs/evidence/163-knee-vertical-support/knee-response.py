from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from local_stand_model import LocalStandModel
from contact_wrench import contact_wrench
source='sim/rl/runs/kl005_lr3e5_update64k_20261009'
candidate=next(x for x in map(json.loads,Path('docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1)
trials=[]
for name in ['stand_early_capture','stand_early_capture_pitch_only']:
 trials += [json.loads(x) for x in Path('sim/rl/runs/screen_logs/'+name+'.diagnostic.jsonl').read_text().splitlines()]
results=[]
for mode in ['fixed_baseline','paired_pitch_only']:
 for seed in [83002,86013]:
  r=next(x for x in trials if x['candidate']==mode and x['seed']==seed)
  for t in [.1,.2,.24,.3]:
   for pulse in [0.,-.01,.01]:
    p=LocalStandModel(source,candidate);e=p.env;e.init_seed=seed;e.reset();e.command=np.zeros(3);e.resample_steps=0
    try:
     for row in r['rows'][:round(t/.02)]:obs,_,fell,_,info=e.step(np.array(row['action']))
     expected=r['rows'][round(t/.02)-1];assert e.data.xpos[e.base][:2].tolist()==expected['position_xy']
     before=e.data.xpos[e.base].copy();loaded_before={k:bool(v) for k,v in e.loaded.items()}
     action=np.array(r['rows'][round(t/.02)]['action']);action[[2,5]]+=pulse
     obs,_,fell,_,info=e.step(action)
     vb=obs[9:12]/.1+np.cross(obs[:3]/.25,e.model.body_pos[e.base]);vz=-float(np.dot(obs[3:6],vb))
     _,_,cs=contact_wrench(e.model,e.data,e.root);f=np.zeros(2)
     for c in cs:
      b=e.model.body(c['body']).name
      if any(s in b for s in ['foot','toe','metatarsus']):
       for i,side in enumerate(['left','right']):
        if b.startswith(side):f[i]+=max(0.,c['force_world_N'][2])
     if pulse==0.:
      nex=r['rows'][round(t/.02)];assert e.data.xpos[e.base][:2].tolist()==nex['position_xy'] and float(info['tilt'])==nex['tilt']
     results.append(dict(mode=mode,seed=seed,time=t,knee_pulse_normalized=pulse,body_point_vz_world_estimate_mps=vz,height=float(e.data.xpos[e.base][2]),height_before=float(before[2]),pitch_rate=float(obs[1]/.25),foot_load_N=f.tolist(),loaded_before=loaded_before,flight=bool(info['flight']),fell=bool(fell),actual_motor_torque=e.data.actuator_force[e.act[e.policy_idx]].tolist(),offline_replay_from_original_start=True))
    finally:p.close()
Path('sim/rl/runs/screen_logs/knee_vertical_response.json').write_text(json.dumps(results,indent=2)+'\n')
for mode in ['fixed_baseline','paired_pitch_only']:
 for seed in [83002,86013]:
  for t in [.1,.2,.24,.3]:
   rr=[x for x in results if x['mode']==mode and x['seed']==seed and x['time']==t];b=next(x for x in rr if x['knee_pulse_normalized']==0.)
   print(mode,seed,t,'basevz',round(b['body_point_vz_world_estimate_mps'],4),[(x['knee_pulse_normalized'],round(x['body_point_vz_world_estimate_mps']-b['body_point_vz_world_estimate_mps'],5),np.round(x['foot_load_N'],2).tolist(),x['flight']) for x in rr],flush=True)
