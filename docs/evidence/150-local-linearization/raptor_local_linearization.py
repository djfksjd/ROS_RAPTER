from pathlib import Path
import sys,json
import numpy as np
import mujoco
sys.path.insert(0,str(Path('sim/rl').resolve()))
from run_env import RunEnv
from train_run import env_kwargs

class LocalPlant:
 def __init__(self):
  source=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009');self.kw=env_kwargs(json.loads((source/'args.json').read_text()));self.env=RunEnv(**{**self.kw,'episode_s':65.,'level':0.},randomize=False,seed=7);self.env.reset()
  candidates=list(map(json.loads,Path('sim/rl/runs/screen_logs/static_equilibrium_margin.diagnostic.jsonl').read_text().splitlines()));self.candidate=next(x for x in candidates if x['start_knee']==1.1);assert self.candidate['static_feasible']
  self.q=np.array(self.candidate['qpos']);self.u=np.array(self.candidate['action']);self.nv=self.env.model.nv;self.na=len(self.u);self.nx=2*self.nv+self.na
 def restore(self,x):
  e=self.env;m,d=e.model,e.data;e.reset();e.command=np.zeros(3);e.resample_steps=0;d.qpos[:]=self.q;mujoco.mj_integratePos(m,d.qpos,x[:self.nv],1.);d.qvel[:]=x[self.nv:2*self.nv];d.qacc[:]=0.;d.qacc_warmstart[:]=0.;d.qfrc_applied[:]=0.;d.xfrc_applied[:]=0.
  prev=e.q0.copy();prev[e.policy_idx]=e.q0[e.policy_idx]+e.scale[e.policy_idx]*(self.u+x[2*self.nv:]);e.ctrl_prev=prev.copy();e.last_action=self.u+x[2*self.nv:];d.ctrl[e.act]=prev;e._clamp();mujoco.mj_forward(m,d)
 def state(self):
  e=self.env;m,d=e.model,e.data;dq=np.zeros(self.nv);mujoco.mj_differentiatePos(m,dq,1.,self.q,d.qpos)
  previous=(e.ctrl_prev[e.policy_idx]-e.q0[e.policy_idx])/e.scale[e.policy_idx]-self.u
  return np.r_[dq,d.qvel.copy(),previous]
 def transition(self,x,u):
  self.restore(x);r=self.env.step(self.u+u)
  assert not r[2] and all(r[4]['loaded'].values()),'left local double-support regime'
  return self.state()
 def close(self):self.env.close()

def main():
 plant=LocalPlant();out=Path('sim/rl/runs/screen_logs/local_equilibrium_linearization');out.mkdir(exist_ok=False)
 try:
  zero=np.zeros(plant.nx);uz=np.zeros(plant.na);base=plant.transition(zero,uz);summary=[];previous=None
  for epsilon in [1e-5,1e-6]:
   A=np.empty((plant.nx,plant.nx));B=np.empty((plant.nx,plant.na))
   for j in range(plant.nx):
    delta=zero.copy();delta[j]=epsilon;A[:,j]=(plant.transition(delta,uz)-plant.transition(-delta,uz))/(2*epsilon)
   for j in range(plant.na):
    delta=uz.copy();delta[j]=epsilon;B[:,j]=(plant.transition(zero,delta)-plant.transition(zero,-delta))/(2*epsilon)
   assert np.array_equal(base,plant.transition(zero,uz)),'map is history dependent despite reset'
   vals=np.linalg.eigvals(A);unstable=[]
   for v in vals:
    if abs(v)>1.0001:
     sv=np.linalg.svd(np.c_[v*np.eye(plant.nx)-A,B],compute_uv=False);unstable.append(dict(real=float(v.real),imag=float(v.imag),magnitude=float(abs(v)),pbh_min_singular=float(sv[-1])))
   np.savez(out/('epsilon_'+str(epsilon)+'.npz'),A=A,B=B,baseline_step=base,q_reference=plant.q,u_reference=plant.u)
   rec=dict(epsilon=epsilon,nv=plant.nv,nx=plant.nx,active_controls=plant.na,unstable_modes=unstable,baseline_step_norm=float(np.linalg.norm(base)),repeat_exact=True)
   if previous is not None:rec['relative_A_change']=float(np.linalg.norm(A-previous[0])/np.linalg.norm(previous[0]));rec['relative_B_change']=float(np.linalg.norm(B-previous[1])/np.linalg.norm(previous[1]))
   summary.append(rec);previous=(A,B);print(rec,flush=True)
  (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 finally:plant.close()
if __name__=='__main__':main()
