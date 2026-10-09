from pathlib import Path
import sys,json
import numpy as np
import torch
sys.path.insert(0,str(Path('sim/rl').resolve()))
import train_symmetric
from stable_baselines3 import PPO
class Captured(Exception):pass
out=Path('docs/evidence/127-first-rollout-advantages');out.mkdir(exist_ok=False)
def capture(self):
 b=self.rollout_buffer
 assert b.full and b.pos==1024 and b.n_envs==4 and self._n_updates==0
 arrays={k:getattr(b,k).copy() for k in ['observations','actions','rewards','values','returns','advantages','episode_starts','log_probs']}
 np.savez_compressed(out/'rollout.npz',**arrays)
 adv=arrays['advantages'].reshape(-1);z=(adv-adv.mean())/(adv.std()+1e-8);o=arrays['observations'].reshape(-1,46);v=arrays['values'].reshape(-1);ret=arrays['returns'].reshape(-1)
 cx=o[:,6]*10;cy=o[:,7]/2;wy=o[:,8]*2;gyro=o[:,2]*4;vx=o[:,9]*10;vy=o[:,10]*10
 masks={'all':np.ones(len(o),bool),'zero_command':(cx==0)&(cy==0)&(wy==0),'moving_turn':(cx>=3)&(np.abs(wy)>=.5),'moving_near_straight':(cx>=3)&(np.abs(wy)<.1),'target_speed_turn':(cx>=3.5)&(cx<=4.5)&(np.abs(wy)>=.5)}
 stats={}
 for name,mask in masks.items():
  n=int(mask.sum())
  stats[name]={'samples':n}
  if n:
   stats[name].update(value_mean=float(v[mask].mean()),gae_return_mean=float(ret[mask].mean()),advantage_mean=float(adv[mask].mean()),standardized_positive_fraction=float(np.mean(z[mask]>0)),yaw_error_mean=float(np.abs(wy[mask]-gyro[mask]).mean()),vx_mean=float(vx[mask].mean()),abs_vy_mean=float(np.abs(vy[mask]).mean()))
 stats['reward_units']='VecNormalize training normalized rewards and GAE returns; not raw evaluation reward or full Monte Carlo return'
 stats['critic_explained_variance_vs_gae']=float(1-np.var(ret-v)/np.var(ret));stats['total_samples']=len(o);stats['optimizer_updates']=self._n_updates
 (out/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n');self.save(out/'unchanged_model.zip');print(json.dumps(stats,ensure_ascii=False),flush=True)
 raise Captured()
if __name__=='__main__':
 torch.set_num_threads(1);original=PPO.train;PPO.train=capture
 sys.argv=['train_symmetric.py','sim/rl/runs/sym_init_20261009','--name','first_rollout_diagnostic_20261009','--steps','4096','--seed','0','--target-kl','.005']
 try:
  try:train_symmetric.main()
  except Captured:print('Stopped intentionally before first optimizer update; diagnostic complete',flush=True)
 finally:PPO.train=original
