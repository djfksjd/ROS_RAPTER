"""No learning: expand the real stand checkpoint, compare actor/critic, smoke-test the quick gate."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'sim/rl'))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from run_env import RunEnv
from train_run import env_kwargs, expand_inputs
from mirror_probe import ObservationReflection
from sym_policy import SymPolicy,freeze_obs,reflection_constants
from launch_j1 import plan,INIT,E


def initialized():
    args=plan();probe=RunEnv(**env_kwargs(args),randomize=False,seed=0)
    vn,plain=expand_inputs(str(INIT/'model.zip'),DummyVecEnv([lambda:probe]),1.)
    freeze_obs(vn)
    r=ObservationReflection(probe.active,probe.policy_idx,probe.q0,probe.obs_vel,probe.obs_contact,
                            len(probe.scan_dx),bool(probe.obs_hold),True)
    sym=reflection_constants(r,vn)
    model=PPO(SymPolicy,vn,n_steps=plain.n_steps,batch_size=plain.batch_size,n_epochs=plain.n_epochs,
              learning_rate=plain.learning_rate,gamma=plain.gamma,gae_lambda=plain.gae_lambda,clip_range=plain.clip_range,
              ent_coef=plain.ent_coef,max_grad_norm=plain.max_grad_norm,device='cpu',seed=0,
              policy_kwargs={**plain.policy_kwargs,'sym':sym})
    model.policy.load_state_dict(plain.policy.state_dict(),strict=False);model.num_timesteps=plain.num_timesteps
    return model,vn,probe


def compare(model,vn):
    old=PPO.load(str(INIT/'model.zip'),device='cpu')
    rng=np.random.default_rng(19905);obs=rng.uniform(-5.,5.,(4096,55)).astype(np.float32)
    max_mean=max_value=0.
    with torch.no_grad():
        source=torch.from_numpy(obs)
        mean=old.policy.get_distribution(source).distribution.mean;value=old.policy.predict_values(source)
        for request in (0.,1.):
            expanded=torch.from_numpy(np.c_[obs,np.full(len(obs),request,dtype=np.float32)])
            new_mean=model.policy.get_distribution(expanded).distribution.mean;new_value=model.policy.predict_values(expanded)
            max_mean=max(max_mean,float(torch.max(torch.abs(new_mean-mean))));max_value=max(max_value,float(torch.max(torch.abs(new_value-value))))
            torch.testing.assert_close(new_mean,mean,rtol=1e-5,atol=2e-6)
            torch.testing.assert_close(new_value,value,rtol=1e-5,atol=2e-6)
        torch.testing.assert_close(old.policy.log_std,model.policy.log_std,rtol=0,atol=0)
        for key,w in model.policy.state_dict().items():
            if key.endswith('.0.weight') and w.ndim==2 and w.shape[1]==56:
                assert torch.count_nonzero(w[:,-1])==0
    assert vn.obs_rms.mean[-1]==0 and vn.obs_rms.var[-1]==1 and vn.norm_reward
    return dict(observations=8192,actor_mean_max_abs_error=max_mean,critic_max_abs_error=max_value,
                input_shape=[55,56],new_input_mean=0,new_input_std=1,action_std_preserved=True,
                norm_reward=vn.norm_reward,return_std=float(vn.ret_rms.var**.5),training_steps=0)


def main():
    import stage_gates as sg
    from gate_judge import judge
    torch.set_num_threads(1)
    model,vn,probe=initialized();result=compare(model,vn)
    with tempfile.TemporaryDirectory(prefix='raptor-j1-init-') as tmp:
        d=Path(tmp)/'j1_untrained_init';d.mkdir()
        model.save(d/'model.zip');vn.save(str(d/'vecnorm.pkl'));(d/'args.json').write_text(json.dumps(plan()))
        out=E/'eval/j1_prepare_ready';out.mkdir(exist_ok=True)
        rows=[]
        with (out/'quick_init.jsonl').open('x') as f:
            for row in sg.jump(d,True,ROOT/'sim/rl/runs/l80_w2',trace_dir=out/'raw'):
                row={'gate':'jump','run':d.name,**row};f.write(json.dumps(row)+'\n');f.flush();rows.append(row)
        (out/'quick_init_judge.txt').write_text('\n'.join(str(judge(r)) for r in rows)+'\nQUICK untrained smoke only; not adoption\n')
        result['quick_episodes']=sum(r['n'] for r in rows)
        result['quick_requested']=sum(e['request_time_s'] is not None for r in rows for e in r['episodes'])
        result['quick_successes']=sum(e['success_time_s'] is not None and e['t_fall'] is None for r in rows for e in r['episodes'])
    vn.close();(E/'eval/j1_init_compat.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
