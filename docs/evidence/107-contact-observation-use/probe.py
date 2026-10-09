"""Read-only actor sensitivity to two contact flags on actual reached states.
Shadow predictions never control physics. Not a causal task-improvement test.
"""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load

def probe(run,yaw,shadow):
 kw=env_kwargs(json.loads((Path(run)/'args.json').read_text()))
 env=RunEnv(**{**kw,'episode_s':15.,'level':0.},randomize=False,seed=7);env.init_seed=52001
 model,norm=load(str(Path(run)/'model.zip'),env);rows=[];controls=[]
 try:
  norm.reset();env.command=np.array([4.,0.,0.]);env.resample_steps=0
  assert env.obs_contact and env.observation_space.shape==(48,)
  for i in range(500):
   t=round(float(env.data.time),12);env.command[2]=yaw if 4.<=t<6. else 0.
   raw=env._obs()[None];action,_=model.predict(norm.normalize_obs(raw),deterministic=True)
   if shadow:
    alternate=raw.copy();alternate[0,-2:]=1-raw[0,-2:]
    alt,_=model.predict(norm.normalize_obs(alternate),deterministic=True)
    rows.append(dict(t=t,loaded=raw[0,-2:].tolist(),action=action[0].tolist(),flipped_action=alt[0].tolist(),max_action_difference=float(np.max(np.abs(action-alt)))))
   _,_,done,_,_=env.step(action[0]);controls.append(dict(action=action[0].tolist(),qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist()))
   if done:break
  if not shadow:return controls
  state=model.policy.state_dict();weights={k:float(v[:,-2:].abs().max()) for k,v in state.items() if k in ('mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight')}
  return dict(run=run,yaw=yaw,seed=52001,fell=bool(done),weights_max_abs_new_columns=weights,rows=rows,controls=controls)
 finally:norm.close()
if __name__=='__main__':
 run='sim/rl/runs/contact_obs_train_contact_20261009';p=Path('docs/evidence/107-contact-observation-use');p.mkdir(exist_ok=True)
 with (p/'probe.jsonl').open('x') as f:
  for yaw in (-1.,0.,1.):
   plain=probe(run,yaw,False);out=probe(run,yaw,True);assert plain==out['controls'];out['shadow_control_parity_exact']=True
   f.write(json.dumps(out,allow_nan=False)+'\n');f.flush()
   active=[r for r in out['rows'] if 4<=r['t']<6]
   print(json.dumps(dict(yaw=yaw,rows=len(out['rows']),turn_samples=len(active),max_action_difference=max(r['max_action_difference'] for r in active),mean_action_difference=np.mean([r['max_action_difference'] for r in active]),weights=out['weights_max_abs_new_columns'],parity=True)),flush=True)
