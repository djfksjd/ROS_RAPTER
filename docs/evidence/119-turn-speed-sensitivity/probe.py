import sys,json,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load
from yaw_feedback import heading_rate
runs=['kl005_update64k_20261009','kl005_linerr2_update64k_20261009']
out=Path('sim/rl/runs/screen_logs/kl_speed_sensitivity.jsonl')
with out.open('x') as f:
 for name in runs:
  for yaw in [-1.,1.]:
   trajectories=[]
   for enabled in [False,True]:
    run=Path('sim/rl/runs')/name;kw=env_kwargs(json.loads((run/'args.json').read_text()))
    env=RunEnv(**{**kw,'episode_s':15.,'level':0.},randomize=False,seed=7);env.init_seed=70001
    model,norm=load(str(run/'model.zip'),env);norm.reset();env.resample_steps=0
    rows=[];traj=[]
    try:
     for i in range(500):
      time=i*.02;env.command=np.array([4.,0.,yaw if 4.<=time<6. else 0.])
      obs=norm.normalize_obs(env._obs()[None]);a,_=model.predict(obs,deterministic=True)
      pred={}
      if enabled and 3.8<=time<=6.5:
       for vx in [4.4,4.8]:
        env.command[0]=vx;alt,_=model.predict(norm.normalize_obs(env._obs()[None]),deterministic=True);pred[str(vx)]=alt[0].tolist()
       env.command[0]=4.
      _,_,fell,_,_=env.step(a[0]);traj.append(np.r_[env.data.qpos.copy(),env.data.qvel.copy()])
      if 3.8<=time<=6.5:
       rows.append(dict(time=time,turn_active=4.<=time<6.,vx=float(env.body_velocity(env.base)[1][0]),heading_rate=heading_rate(env.data.xmat[env.base],env.body_velocity(env.base)[0]),action=a[0].tolist(),counterfactual_speed_actions=pred))
      if fell:break
     trajectories.append(np.array(traj));turn=[r for r in rows if r['turn_active']]
     record=dict(run=name,yaw=yaw,enabled=enabled,seed=70001,fell=bool(fell),min_vx=min(r['vx'] for r in turn),max_vx=max(r['vx'] for r in turn),mean_vx=float(np.mean([r['vx'] for r in turn])),min_vx_time=min(turn,key=lambda r:r['vx'])['time'],rows=rows)
     if enabled:
      assert np.array_equal(*trajectories),'counterfactual probe changed actual physics'
      record['trajectory_exactly_matches_disabled']=True
      record['speed_input_mean_abs_action_change']={str(vx):float(np.mean([np.abs(np.array(r['counterfactual_speed_actions'][str(vx)])-np.array(r['action'])).mean() for r in turn])) for vx in [4.4,4.8]}
     f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();print({k:v for k,v in record.items() if k!='rows'},flush=True)
    finally:norm.close()
