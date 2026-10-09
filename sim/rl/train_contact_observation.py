"""Isolated contact-observation ablation, preserving 8 actions and all physical limits."""
import argparse,copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor,VecNormalize,DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback,CallbackList
from stable_baselines3.common.logger import configure
from run_env import RunEnv
from train_run import env_kwargs,make
from command_sequence import CommandSequenceEnv
from symmetric_policy import SymmetricPolicy,configuration
from train_symmetric import EpisodeLog

HERE=Path(__file__).resolve().parent
INPUT_WEIGHTS={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}


def transfer_policy(old,new,extra):
    if extra not in (0,2):raise ValueError('only the two appended contact features may be added')
    state={k:v.clone() for k,v in old.state_dict().items() if not k.startswith('reflection_')}
    target=new.state_dict()
    for key,value in list(state.items()):
        if value.shape!=target[key].shape:
            if key not in INPUT_WEIGHTS or target[key].shape!=(value.shape[0],value.shape[1]+extra):
                raise ValueError(f'unexpected shape change: {key}')
            padded=torch.zeros_like(target[key]);padded[:,:value.shape[1]]=value;state[key]=padded
    missing,unexpected=new.load_state_dict(state,strict=False)
    if unexpected or any(not k.startswith('reflection_') for k in missing):raise ValueError('incomplete policy transfer')
    new.optimizer.load_state_dict(copy.deepcopy(old.optimizer.state_dict()))
    old_params=dict(old.named_parameters())
    for key,param in new.named_parameters():
        old_shape=old_params[key].shape
        if old_shape!=param.shape:
            for field in ['exp_avg','exp_avg_sq','max_exp_avg_sq']:
                if field in new.optimizer.state[param]:
                    value=new.optimizer.state[param][field]
                    assert value.shape==old_shape
                    padded=torch.zeros_like(param);padded[:,:old_shape[1]]=value
                    new.optimizer.state[param][field]=padded


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--name',required=True)
    p.add_argument('--contact',action='store_true');p.add_argument('--steps',type=int,default=500000)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--initialize-only',action='store_true');a=p.parse_args()
    if a.steps<=0:p.error('positive steps required')
    source=Path(a.source).resolve();out=HERE/'runs'/a.name
    torch.set_num_threads(1)
    args=json.loads((source/'args.json').read_text());oldkw=env_kwargs(args)
    if oldkw['obs_contact']:p.error('this controlled ablation requires a 46-feature source')
    oldprobe=RunEnv(**oldkw,randomize=False,seed=7)
    oldnorm=VecNormalize.load(str(source/'vecnorm.pkl'),DummyVecEnv([lambda:oldprobe]))
    old=PPO.load(source/'model.zip',device='cpu')
    if not isinstance(old.policy,SymmetricPolicy):raise ValueError('requires verified SymmetricPolicy source')
    extra=2*int(a.contact);kw={**oldkw,'obs_contact':a.contact,'episode_s':22.}
    probe=RunEnv(**kw,randomize=False,seed=7)
    mean=np.r_[old.policy.reflection_mean.cpu().numpy(),np.zeros(extra)]
    std=np.r_[old.policy.reflection_std.cpu().numpy(),np.ones(extra)]
    stats=SimpleNamespace(obs_rms=SimpleNamespace(mean=mean,var=std**2),epsilon=0.,clip_obs=old.policy.input_clip)
    config=configuration(probe,stats,old.policy.yaw_scale)
    env=VecMonitor(SubprocVecEnv([make(kw,a.seed*100+i,CommandSequenceEnv) for i in range(4)]),info_keywords=('sequence_kind',))
    # Construct matching spaces while retaining the source reward statistics.
    norm=VecNormalize(env,norm_obs=True,norm_reward=True,gamma=oldnorm.gamma,clip_reward=oldnorm.clip_reward,epsilon=oldnorm.epsilon)
    norm.norm_obs=False;norm.ret_rms=copy.deepcopy(oldnorm.ret_rms)
    norm.obs_rms.mean=mean.copy();norm.obs_rms.var=np.maximum(0.,std**2-norm.epsilon)
    model=PPO(SymmetricPolicy,norm,n_steps=1024,batch_size=4096,n_epochs=5,learning_rate=3e-4,
              gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.003,max_grad_norm=1.,device='cpu',
              policy_kwargs={**old.policy_kwargs,'reflection_config':config},seed=a.seed,verbose=0)
    transfer_policy(old.policy,model.policy,extra);model.num_timesteps=old.num_timesteps
    # Same action-noise RNG starting point in both sides of this new comparison.
    torch.manual_seed(a.seed)
    out.mkdir(exist_ok=False);args.update(name=a.name,init=str(source/'model.zip'),seed=a.seed,steps=a.steps,
       obs_contact=a.contact,episode_s=22.,command_sequence=True,sequence_control=False,
       fixed_input_normalization=True,raw_policy_observation=True,symmetry_policy=True,
       symmetry_yaw_scale=old.policy.yaw_scale,training_noise_rng_reset=True)
    (out/'args.json').write_text(json.dumps(args,indent=2)+'\n')
    (out/'reflection-config.json').write_text(json.dumps(config,indent=2)+'\n')
    snapshots=out/'source-code';snapshots.mkdir()
    for name in ['train_contact_observation.py','train_symmetric.py','run_env.py','train_run.py','command_sequence.py','symmetric_policy.py','mirror_probe.py','raptor_env.py']:
        (snapshots/name).write_bytes((HERE/name).read_bytes())
    model.set_logger(configure(str(out),['csv']));model.save(out/'model_initialized.zip');norm.save(out/'vecnorm_initialized.pkl')
    try:
        if not a.initialize_only:
            cb=CheckpointCallback(save_freq=62500,save_path=str(out/'checkpoints'),name_prefix='contact',save_vecnormalize=True)
            model.learn(total_timesteps=a.steps,reset_num_timesteps=False,callback=CallbackList([cb,EpisodeLog(out/'episodes.jsonl')]))
        model.save(out/'model.zip');norm.save(out/'vecnorm.pkl')
        (out/'completion.json').write_text(json.dumps(dict(initial_steps=old.num_timesteps,final_steps=model.num_timesteps,initialize_only=a.initialize_only),indent=2)+'\n')
    finally:norm.close();oldnorm.close();probe.close()

if __name__=='__main__':main()
