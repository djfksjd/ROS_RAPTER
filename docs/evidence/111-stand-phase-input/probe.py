"""Frozen-policy phase-input ablation. Simulator clock and rewards unchanged."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load
from motion_transition import summarize
run='sim/rl/runs/stand_support_control_20261009'
def probe(phase):
 env=RunEnv(**{**env_kwargs(json.loads((Path(run)/'args.json').read_text())),'episode_s':65.,'level':0.},randomize=False,seed=7);env.init_seed=59001
 model,norm=load(str(Path(run)/'model.zip'),env);rows=[]
 try:
  norm.reset();env.command=np.zeros(3);env.resample_steps=0
  assert not norm.norm_obs and not env.obs_contact and env.observation_space.shape==(46,)
  for k in range(3000):
   raw=env._obs()[None].copy()
   np.testing.assert_allclose(raw[0,-2:],[np.sin(2*np.pi*env.phase),np.cos(2*np.pi*env.phase)],atol=1e-7)
   if phase is not None:raw[0,-2:]=[np.sin(2*np.pi*phase),np.cos(2*np.pi*phase)]
   act,_=model.predict(raw,deterministic=True);before=env.data.xpos[env.base,:2].copy()
   _,_,fell,_,info=env.step(act[0])
   rows.append(dict(t=(k+1)*.02,command_vx=0.,stage='stand',cycle=0,vx=float(info['v_body'][0]),speed_xy=float(np.linalg.norm(info['v_body'][:2])),position_xy=env.data.xpos[env.base,:2].tolist(),before_position_xy=before.tolist(),tilt=float(info['tilt']),loaded={s:bool(v) for s,v in info['loaded'].items()},flight=bool(info['flight']),physical_phase=float(info['phase'])))
   if fell:break
  return dict(run=run,seed=59001,fixed_observation_phase=phase,physical_phase_unmodified=True,fell=bool(fell),summary=summarize(rows,'stand',bool(fell),60.),rows=rows)
 finally:norm.close()
p=Path('docs/evidence/111-stand-phase-input');p.mkdir(exist_ok=True)
with (p/'raw.jsonl').open('x') as f:
 for phase in (None,0.,.25,.5,.75):
  r=probe(phase);f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();print(json.dumps({k:v for k,v in r.items() if k!='rows'}),flush=True)
