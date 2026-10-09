"""Train a separate stand-only PPO skill on the original eight-axis mechanics."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stand_env import StandEnv
from train_run import env_kwargs, make


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--name', required=True)
    parser.add_argument('--steps', type=int, default=503808)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0 or Path(args.name).name != args.name:
        parser.error('positive steps and a single directory name required')
    torch.set_num_threads(1)
    source = Path(args.source).resolve()
    config = json.loads((source/'args.json').read_text())
    if not config.get('arch8') or config.get('arch12'):
        parser.error('original eight-axis source required')
    out = Path(__file__).resolve().parent/'runs'/args.name
    out.mkdir(exist_ok=False)
    config.update(name=args.name, source=str(source), steps=args.steps, seed=args.seed,
                  stand_task=True, stand_reward_version=1, envs=4, episode_s=8.,
                  cmd=[0., 0., 0.], cmd_final=[0., 0., 0.], zero_cmd=1., top_cmd=0.,
                  disturb_items=[], stand_policy_initialized_from_scratch=True,
                  init=None, symmetry_policy=False, symmetry_yaw_scale=None,
                  raw_policy_observation=False, fixed_input_normalization=False,
                  learning_rate=3e-4, target_kl=.01, initial_log_std=-2.,
                  policy_architecture={'pi':[128,128], 'vf':[128,128]})
    (out/'args.json').write_text(json.dumps(config, indent=2)+'\n')
    kwargs = env_kwargs(config)
    vector = VecNormalize(VecMonitor(SubprocVecEnv(
        [make(kwargs, args.seed+i, cls=StandEnv) for i in range(4)], start_method='spawn')),
        norm_obs=True, norm_reward=True, clip_obs=10.)
    try:
        model = PPO('MlpPolicy', vector, seed=args.seed, device='cpu', n_steps=1024,
                    batch_size=4096, n_epochs=5, learning_rate=3e-4, gamma=.99,
                    gae_lambda=.95, clip_range=.2, ent_coef=.003, max_grad_norm=1.,
                    target_kl=.01, policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128]),
                                                    log_std_init=-2.))
        model.set_logger(configure(str(out), ['stdout','csv']))
        model.save(out/'initial_model.zip')
        vector.save(out/'initial_vecnorm.pkl')
        model.learn(args.steps, callback=CheckpointCallback(
            save_freq=16384, save_path=str(out/'checkpoints'),
            save_vecnormalize=True, name_prefix='stand'))
        model.save(out/'model.zip')
        vector.save(out/'vecnorm.pkl')
        (out/'completion.json').write_text(json.dumps(
            dict(status='training_complete', actual_steps=model.num_timesteps,
                 evaluation_pass_unproven=True), indent=2)+'\n')
    finally:
        vector.close()


if __name__ == '__main__':
    main()
