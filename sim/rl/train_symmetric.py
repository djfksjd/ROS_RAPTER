"""Isolated PPO candidate, initialized from the verified reflection inference actor."""
import argparse
import json
import math
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor,VecNormalize,DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback,CallbackList,BaseCallback
from stable_baselines3.common.logger import configure
from run_env import RunEnv
from train_run import env_kwargs,make
from symmetric_policy import SymmetricPolicy,configuration

HERE=Path(__file__).resolve().parent

def set_turn_exploration(policy,env,std):
    """Change only paired hip-roll and tail-yaw action noise, preserving other axes."""
    if not math.isfinite(std) or std<=0:raise ValueError('positive finite std required')
    names=[env.active[i] for i in env.policy_idx]
    slots=[i for i,n in enumerate(names) if n.endswith(('hip_roll_joint','tail_yaw_joint'))]
    if len(slots)!=3:raise ValueError('expected two hip-roll and one tail-yaw action')
    with torch.no_grad():policy.log_std[slots]=math.log(std)
    return slots

class EpisodeLog(BaseCallback):
    def __init__(self,path):
        super().__init__();self.path=path
    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                e=info['episode']
                row={'steps':self.num_timesteps,'reward':float(e['r']),'length':int(e['l']),
                     'elapsed_s':float(e['t']),'sequence_kind':e.get('sequence_kind')}
                with self.path.open('a') as f:f.write(json.dumps(row)+'\n')
        return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--name',required=True)
    p.add_argument('--steps',type=int,default=500000);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--command-sequence',action='store_true')
    p.add_argument('--sequence-control',action='store_true')
    p.add_argument('--sequence-turn-seconds',type=float,default=2.)
    p.add_argument('--lin-error-weight',type=float,default=None)
    p.add_argument('--lateral-error-weight',type=float,default=None)
    p.add_argument('--stand-support-weight',type=float,default=None)
    p.add_argument('--turn-exploration-std',type=float,default=None)
    p.add_argument('--target-kl',type=float,default=None,
                   help='Stop remaining PPO epochs after excessive measured KL; default disables this guard')
    p.add_argument('--reset-optimizer',action='store_true',
                   help='Keep initialized policy weights but start with fresh optimizer state')
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--yaw-scale',type=float,default=6.4);p.add_argument('--initialize-only',action='store_true')
    a=p.parse_args();source=Path(a.source).resolve();out=HERE/'runs'/a.name
    if a.steps<=0 or not 0<a.yaw_scale<=8:p.error('positive steps and yaw scale within (0,8] required')
    if a.command_sequence and a.sequence_control:p.error('choose one sequence mode')
    if not math.isfinite(a.sequence_turn_seconds) or not 0<a.sequence_turn_seconds<=8:
        p.error('sequence turn seconds must be within (0,8]')
    if a.sequence_turn_seconds!=2. and not a.command_sequence:
        p.error('extended turn requires command sequence')
    if a.lin_error_weight is not None and (not math.isfinite(a.lin_error_weight) or a.lin_error_weight>0):
        p.error('linear speed error weight must be finite and nonpositive')
    if a.lateral_error_weight is not None and (not math.isfinite(a.lateral_error_weight) or a.lateral_error_weight>0):
        p.error('lateral speed error weight must be finite and nonpositive')
    if a.stand_support_weight is not None and (not math.isfinite(a.stand_support_weight) or a.stand_support_weight>0):
        p.error('stand support weight must be finite and nonpositive')
    if a.turn_exploration_std is not None and (not math.isfinite(a.turn_exploration_std) or a.turn_exploration_std<=0):
        p.error('turn exploration std must be finite and positive')
    if a.target_kl is not None and (not math.isfinite(a.target_kl) or a.target_kl<=0):
        p.error('target KL must be finite and positive')
    if not math.isfinite(a.learning_rate) or a.learning_rate<=0:
        p.error('learning rate must be finite and positive')
    torch.set_num_threads(1)
    args=json.loads((source/'args.json').read_text())
    if a.lin_error_weight is not None:
        weights=json.loads(args['weights']);weights['lin_err']=a.lin_error_weight
        args['weights']=json.dumps(weights)
    if a.lateral_error_weight is not None:
        weights=json.loads(args['weights']);weights['lateral_err']=a.lateral_error_weight
        args['weights']=json.dumps(weights)
    if a.stand_support_weight is not None:
        weights=json.loads(args['weights']);weights['stand_support']=a.stand_support_weight
        args['weights']=json.dumps(weights)
    kw=env_kwargs(args)
    probe=RunEnv(**kw,randomize=False,seed=7)
    oldnorm=VecNormalize.load(str(source/'vecnorm.pkl'),DummyVecEnv([lambda:probe]))
    old=PPO.load(source/'model.zip',device='cpu')
    config=configuration(probe,oldnorm,a.yaw_scale)
    kwargs={**old.policy_kwargs,'reflection_config':config}
    if a.command_sequence or a.sequence_control:
        from command_sequence import CommandSequenceEnv,SequenceControlEnv
        kw={**kw,'episode_s':22.}
        cls=CommandSequenceEnv if a.command_sequence else SequenceControlEnv
        if a.command_sequence:kw['turn_seconds']=a.sequence_turn_seconds
    else:cls=RunEnv
    env=VecMonitor(SubprocVecEnv([make(kw,a.seed*100+i,cls) for i in range(4)]),
                   info_keywords=('sequence_kind',) if a.command_sequence or a.sequence_control else ())
    norm=VecNormalize.load(str(source/'vecnorm.pkl'),env)
    norm.norm_obs=False;norm.training=True;norm.norm_reward=True
    model=PPO(SymmetricPolicy,norm,n_steps=1024,batch_size=4096,n_epochs=5,learning_rate=a.learning_rate,
              gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.003,max_grad_norm=1.,
              device='cpu',policy_kwargs=kwargs,seed=a.seed,verbose=0,target_kl=a.target_kl)
    missing,unexpected=model.policy.load_state_dict(old.policy.state_dict(),strict=False)
    if unexpected or any(not n.startswith('reflection_') for n in missing):
        raise RuntimeError(f'unexpected initialization mismatch: {missing}, {unexpected}')
    if not a.reset_optimizer:
        model.policy.optimizer.load_state_dict(old.policy.optimizer.state_dict())
    exploration_slots=set_turn_exploration(model.policy,probe,a.turn_exploration_std) if a.turn_exploration_std is not None else []
    model.num_timesteps=old.num_timesteps
    out.mkdir(exist_ok=False)
    args.update(name=a.name,init=str(source/'model.zip'),seed=a.seed,steps=a.steps,
                symmetry_policy=True,symmetry_yaw_scale=a.yaw_scale,raw_policy_observation=True,
                fixed_input_normalization=True,command_sequence=a.command_sequence,sequence_control=a.sequence_control,episode_s=kw["episode_s"])
    args.update(turn_exploration_std=a.turn_exploration_std,turn_exploration_slots=exploration_slots)
    args['sequence_turn_seconds']=a.sequence_turn_seconds
    args['target_kl']=a.target_kl
    args['reset_optimizer']=a.reset_optimizer
    args['learning_rate']=a.learning_rate
    code_dir=out/'source-code';code_dir.mkdir()
    for filename in ['train_symmetric.py','symmetric_policy.py','command_sequence.py','run_env.py','train_run.py','raptor_env.py','mirror_probe.py']:
        (code_dir/filename).write_bytes((HERE/filename).read_bytes())
    (out/'args.json').write_text(json.dumps(args,indent=2)+'\n')
    (out/'reflection-config.json').write_text(json.dumps(config,indent=2)+'\n')
    model.set_logger(configure(str(out),['csv']))
    model.save(out/'model_initialized.zip');norm.save(out/'vecnorm_initialized.pkl')
    try:
        if not a.initialize_only:
            cb=CheckpointCallback(save_freq=62500,save_path=str(out/'checkpoints'),name_prefix='sym',save_vecnormalize=True)
            model.learn(total_timesteps=a.steps,reset_num_timesteps=False,callback=CallbackList([cb,EpisodeLog(out/'episodes.jsonl')]))
        model.save(out/'model.zip');norm.save(out/'vecnorm.pkl')
        (out/'completion.json').write_text(json.dumps({'initial_steps':old.num_timesteps,
                                                       'final_steps':model.num_timesteps,
                                                       'initialize_only':a.initialize_only},indent=2)+'\n')
    finally:norm.close();oldnorm.close()

if __name__=='__main__':main()
