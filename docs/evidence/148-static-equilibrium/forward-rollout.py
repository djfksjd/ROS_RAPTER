from pathlib import Path
import sys,json
import numpy as np
import mujoco
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs
from motion_transition import stand_gate
source=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009');kw=env_kwargs(json.loads((source/'args.json').read_text()))
candidates=[x for x in map(json.loads,Path('sim/rl/runs/screen_logs/static_equilibrium_margin.diagnostic.jsonl').read_text().splitlines()) if x['static_feasible']]
def trial(candidate,seed):
 e=RunEnv(**{**kw,'episode_s':15.,'level':0.},randomize=False,seed=7);e.reset();e.command=np.zeros(3);e.resample_steps=0
 try:
  d,m=e.data,e.model;d.qpos[:]=candidate['qpos'];d.qvel[:]=0.;d.qacc[:]=0.;d.qacc_warmstart[:]=0.;d.qfrc_applied[:]=0.;d.xfrc_applied[:]=0.
  if seed is not None:
   rng=np.random.default_rng(seed);d.qpos[e.q_adr[e.policy_idx]]+=rng.normal(0,.001,len(e.policy_idx));d.qvel[0]=rng.uniform(0,.02)
  action=np.array(candidate['action']);target=e.q0.copy();target[e.policy_idx]=e.q0[e.policy_idx]+e.scale[e.policy_idx]*action;d.ctrl[e.act]=target;e.ctrl_prev=target.copy();e.last_action=action.copy();e._clamp();mujoco.mj_forward(m,d)
  initial=dict(qacc=d.qacc.tolist(),torque=d.actuator_force[e.act[e.policy_idx]].tolist(),external_applied_force=d.qfrc_applied.tolist(),external_body_wrench=d.xfrc_applied.tolist())
  rows=[]
  for i in range(500):
   before=d.xpos[e.base][:2].copy();_,_,fell,_,info=e.step(action)
   rows.append(dict(t=(i+1)*.02,position_xy=d.xpos[e.base][:2].tolist(),before_position_xy=before.tolist(),tilt=float(info['tilt']),flight=bool(info['flight']),speed_xy=float(np.linalg.norm(info['v_body'][:2])),height=float(d.xpos[e.base][2]),heat=e.heat_inst.tolist()))
   if fell:break
  heat=np.mean([x['heat'] for x in rows[-2000:]],axis=0);gate=stand_gate(rows)
  return dict(candidate_start_knee=candidate['start_knee'],initial_state_kind='equilibrium_exact' if seed is None else 'equilibrium_microperturbed',seed=seed,initial=initial,action=action.tolist(),seconds=rows[-1]['t'],fell=bool(fell),gate=gate,max_load_average=float(np.max(heat[e.motor_cont<1e8])),rows=rows,catalogue_pass=False,original_nominal_start_tested=False)
 finally:e.close()
p=Path('sim/rl/runs/screen_logs/static_pose_forward.diagnostic.jsonl')
with p.open('x') as f:
 for c in candidates:
  for seed in [None,84001,84002]:
   r=trial(c,seed);f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();print({k:v for k,v in r.items() if k not in ['rows','initial','action']},flush=True)
