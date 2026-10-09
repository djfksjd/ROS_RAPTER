from pathlib import Path
import sys,json
import torch,numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from stable_baselines3 import PPO
from symmetric_policy import SymmetricPolicy
p=Path('docs/evidence/129-first-gradient');p.mkdir(exist_ok=False)
rows=[]
for mode in ['v2','catalogue']:
 run=Path('docs/evidence/127-first-rollout-advantages')/mode;b=np.load(run/'rollout.npz');m=PPO.load(run/'unchanged_model.zip',device='cpu');policy=m.policy
 obs=torch.tensor(b['observations'].reshape(-1,46));actions=torch.tensor(b['actions'].reshape(-1,8));oldlp=torch.tensor(b['log_probs'].reshape(-1));adv=torch.tensor(b['advantages'].reshape(-1));adv=(adv-adv.mean())/(adv.std()+1e-8);ret=torch.tensor(b['returns'].reshape(-1))
 value,lp,entropy=policy.evaluate_actions(obs,actions);ratio=torch.exp(lp-oldlp);ploss=-(adv*ratio).mean();eloss=-m.ent_coef*entropy.mean();vloss=m.vf_coef*torch.nn.functional.mse_loss(value.flatten(),ret)
 named=list(policy.named_parameters());params=[v for k,v in named];names=[k for k,v in named]
 def vector(loss):
  g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
  return {k:(v.detach() if v is not None else torch.zeros_like(param)) for k,param,v in zip(names,params,g)}
 pg,eg,vg=vector(ploss),vector(eloss),vector(vloss)
 actor=[k for k in names if 'policy_net' in k or 'action_net' in k or k=='log_std'];assert actor
 def cat(g):return torch.cat([g[k].reshape(-1).double() for k in actor])
 current=cat(pg)+cat(eg);saved={k:m.policy.optimizer.state[param].get('exp_avg',torch.zeros_like(param)) for k,param in named};prior=cat(saved)
 total=torch.sqrt(sum(((pg[k]+eg[k]+vg[k]).double()**2).sum() for k in names));cos=float(torch.dot(current,prior)/(current.norm()*prior.norm()).clamp_min(1e-12))
 rows.append(dict(mode=mode,samples=len(obs),policy_gradient_actor_norm=float(cat(pg).norm()),entropy_gradient_actor_norm=float(cat(eg).norm()),value_gradient_actor_norm=float(cat(vg).norm()),total_gradient_norm=float(total),global_clip_multiplier=min(1.,float(m.max_grad_norm/(total+1e-6))),old_adam_actor_momentum_norm=float(prior.norm()),cosine_old_momentum_current_actor_gradient=cos,log_probability_replay_max_error=float((lp-oldlp).abs().max()),optimizer_updates=m._n_updates,actor_parameter_names=actor))
 assert m._n_updates==0
(p/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');print(rows)
