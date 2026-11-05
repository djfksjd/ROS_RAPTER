"""No learning. Actual j3 -> j4 input expansion, parity and new-height 0M quick baseline."""
import hashlib
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
from train_run import env_kwargs,expand_inputs,widen_command_inputs,rescale_action_outputs
from mirror_probe import ObservationReflection
from sym_policy import SymPolicy,freeze_obs,reflection_constants
from launch_j4 import plan,INIT,E


def initialized():
    args=plan();probe=RunEnv(**env_kwargs(args),randomize=False,seed=0)
    vn,plain=expand_inputs(str(INIT/'model.zip'),DummyVecEnv([lambda:probe]),1.)
    widen_command_inputs(vn,plain,args['cmd_final'])
    old_args=json.loads((INIT/'args.json').read_text())
    rescale_action_outputs(plain,probe.active,probe.policy_idx,old_args,probe.scale,args['resume_action_std'])
    freeze_obs(vn)
    reflection=ObservationReflection(probe.active,probe.policy_idx,probe.q0,probe.obs_vel,probe.obs_contact,
        len(probe.scan_dx),bool(probe.obs_hold),True,True)
    model=PPO(SymPolicy,vn,n_steps=plain.n_steps,batch_size=plain.batch_size,n_epochs=plain.n_epochs,
        learning_rate=plain.learning_rate,gamma=plain.gamma,gae_lambda=plain.gae_lambda,clip_range=plain.clip_range,
        ent_coef=plain.ent_coef,max_grad_norm=plain.max_grad_norm,device='cpu',seed=args['seed'],
        policy_kwargs={**plain.policy_kwargs,'sym':reflection_constants(reflection,vn)})
    model.policy.load_state_dict(plain.policy.state_dict(),strict=False);model.num_timesteps=plain.num_timesteps
    return model,vn,probe


def compare(model,vn):
    import pickle
    old=PPO.load(str(INIT/'model.zip'),device='cpu')
    with (INIT/'vecnorm.pkl').open('rb') as f:old_vn=pickle.load(f)
    np.testing.assert_array_equal(vn.obs_rms.mean[:-1],old_vn.obs_rms.mean)
    np.testing.assert_array_equal(vn.obs_rms.var[:-1],old_vn.obs_rms.var)
    assert vn.obs_rms.mean[-1]==0 and vn.obs_rms.var[-1]==1
    assert vn.ret_rms.var==old_vn.ret_rms.var and vn.norm_reward==old_vn.norm_reward
    obs=np.random.default_rng(19954).uniform(-5.,5.,(4096,56)).astype(np.float32)
    maximum_mean=maximum_value=0.
    with torch.no_grad():
        source=torch.from_numpy(obs)
        mean=old.policy.get_distribution(source).distribution.mean;value=old.policy.predict_values(source)
        for phase in (0.,1.):
            extended=torch.from_numpy(np.c_[obs,np.full(len(obs),phase,dtype=np.float32)])
            new_mean=model.policy.get_distribution(extended).distribution.mean;new_value=model.policy.predict_values(extended)
            maximum_mean=max(maximum_mean,float(torch.max(torch.abs(mean-new_mean))))
            maximum_value=max(maximum_value,float(torch.max(torch.abs(value-new_value))))
            torch.testing.assert_close(new_mean,mean,rtol=1e-5,atol=2e-6)
            torch.testing.assert_close(new_value,value,rtol=1e-5,atol=2e-6)
        torch.testing.assert_close(model.policy.log_std,old.policy.log_std,rtol=0,atol=0)
        for key,w in model.policy.state_dict().items():
            if key.endswith('.0.weight') and w.ndim==2 and w.shape[1]==57:assert torch.count_nonzero(w[:,-1])==0
    return dict(observations=8192,input_shape=[56,57],actor_mean_max_abs_error=maximum_mean,
        critic_max_abs_error=maximum_value,phase_mean=0.,phase_std=1.,old_normalization_exact=True,
        action_std_preserved=True,norm_reward=bool(vn.norm_reward),return_std=float(vn.ret_rms.var**.5),
        training_steps=0,optimizer_state='fresh optimizer via existing expand_inputs, no optimization executed')


def main():
    from check_training_j4 import metrics
    from check_training_j1 import write_rows,judge_text
    import stage_gates as sg
    torch.set_num_threads(1);out=E/'eval/j4_prepare';out.mkdir(exist_ok=True)
    if (out/'quick_0M.jsonl').exists():raise FileExistsError('preserve the declared 0M baseline')
    model,vn,_=initialized();result=compare(model,vn)
    with tempfile.TemporaryDirectory(prefix='raptor-j4-init-') as tmp:
        run=Path(tmp)/'j4_untrained_init';run.mkdir()
        model.save(run/'model.zip');vn.save(str(run/'vecnorm.pkl'));(run/'args.json').write_text(json.dumps(plan()))
        rows=write_rows(out/'quick_0M.jsonl',
            ({'run':run.name,**r} for r in sg.jump(run,True,ROOT/'sim/rl/runs/l80_w2',trace_dir=out/'raw',height_target_m=.20)))
        judge_text(out/'quick_0M_judge.txt',rows)
        result['baseline']=metrics(rows)
    vn.close()
    (out/'initialization.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
