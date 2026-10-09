from pathlib import Path
import json,sys,numpy as np,torch
sys.path.insert(0,'sim/rl')
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load
runs=['seq_compare_catalogue_20261009','contact_obs_init_control_20261009','contact_obs_init_contact_20261009']
result=[]
for yaw in [-1.,0.,1.]:
 envs=[];models=[];norms=[]
 for name in runs:
  path=Path('sim/rl/runs')/name;a=json.loads((path/'args.json').read_text())
  e=RunEnv(**{**env_kwargs(a),'level':0.,'episode_s':15.},randomize=False,seed=7);e.init_seed=47001
  m,n=load(str(path/'model.zip'),e);n.reset();e.command=np.array([4.,0.,0.]);e.resample_steps=0
  envs.append(e);models.append(m);norms.append(n)
 action_diff=np.zeros(2);qpos_diff=np.zeros(2);qvel_diff=np.zeros(2);exact=[True,True]
 try:
  for _ in range(500):
   t=round(float(envs[0].data.time),12)
   for e in envs:e.command[2]=yaw if 4.<=t<6. else 0.
   actions=[m.predict(n.normalize_obs(e._obs()[None]),deterministic=True)[0][0] for e,m,n in zip(envs,models,norms)]
   infos=[e.step(a) for e,a in zip(envs,actions)]
   for i in range(2):
    action_diff[i]=max(action_diff[i],float(np.abs(actions[0]-actions[i+1]).max()))
    qpos_diff[i]=max(qpos_diff[i],float(np.abs(envs[0].data.qpos-envs[i+1].data.qpos).max()))
    qvel_diff[i]=max(qvel_diff[i],float(np.abs(envs[0].data.qvel-envs[i+1].data.qvel).max()))
    exact[i]=exact[i] and np.array_equal(actions[0],actions[i+1]) and np.array_equal(envs[0].data.qpos,envs[i+1].data.qpos) and np.array_equal(envs[0].data.qvel,envs[i+1].data.qvel)
   if any(r[2] for r in infos):break
  result.append(dict(yaw=yaw,seed=47001,samples=envs[0].steps,exact=exact,max_action_difference=action_diff.tolist(),max_qpos_vector_difference=qpos_diff.tolist(),max_qvel_vector_difference=qvel_diff.tolist()))
 finally:
  for n in norms:n.close()
Path('/tmp/raptor-contact-initial-parity.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
