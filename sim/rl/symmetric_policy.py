"""Trainable reflection-averaged Gaussian actor over RAW RunEnv observations.

The original normalization is fixed inside the policy; external VecNormalize
must use norm_obs=False. Actor branch inputs and means are clipped as in the
legacy inference average. The critic retains its original normalized input.
This constrains actor mean/noise symmetry, not full robot dynamics or critic.
"""
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from mirror_probe import ObservationReflection


def configuration(env, norm, yaw_scale):
    reflection=ObservationReflection(env.active,env.policy_idx,env.q0,env.obs_vel,getattr(env,'obs_contact',False))
    offset=reflection.observation(np.zeros(reflection.size,dtype=float))
    matrix=(reflection.observation(np.eye(reflection.size))-offset).T
    index=np.argmax(np.abs(matrix),axis=1)
    sign=matrix[np.arange(reflection.size),index]
    assert np.allclose(np.abs(matrix).sum(axis=1),1.)
    return dict(index=index.tolist(),sign=sign.tolist(),offset=offset.tolist(),
                action_index=reflection.action_index.tolist(),action_sign=reflection.action_sign.tolist(),
                mean=norm.obs_rms.mean.tolist(),std=np.sqrt(norm.obs_rms.var+norm.epsilon).tolist(),
                clip=float(norm.clip_obs),yaw_scale=float(yaw_scale))


class SymmetricPolicy(ActorCriticPolicy):
    def __init__(self,*args,reflection_config,**kwargs):
        super().__init__(*args,**kwargs)
        if not isinstance(self.action_dist,DiagGaussianDistribution) or self.squash_output:
            raise ValueError('reflection actor requires unsquashed diagonal Gaussian actions')
        if not np.all(self.action_space.low==-1) or not np.all(self.action_space.high==1):
            raise ValueError('reflection actor requires unit symmetric action bounds')
        self.reflection_config=reflection_config
        for name in ['mean','std','sign','offset','action_sign']:
            self.register_buffer('reflection_'+name,torch.tensor(reflection_config[name],dtype=torch.float64))
        for name in ['index','action_index']:
            self.register_buffer('reflection_'+name,torch.tensor(reflection_config[name],dtype=torch.long))
        self.input_clip=reflection_config['clip'];self.yaw_scale=reflection_config['yaw_scale']

    def _normalize(self,raw):
        return ((raw.double()-self.reflection_mean)/self.reflection_std).clamp(-self.input_clip,self.input_clip).float()

    def mirror(self,raw):
        return raw.double()[...,self.reflection_index]*self.reflection_sign+self.reflection_offset

    def _branch(self,raw):
        features=self.extract_features(self._normalize(raw),self.pi_features_extractor)
        return self.action_net(self.mlp_extractor.forward_actor(features)).clamp(-1,1)

    def mean_action(self,raw):
        scaled=raw.float().clone();scaled[...,8]*=self.yaw_scale
        a=self._branch(scaled);b=self._branch(self.mirror(scaled).float())
        return .5*(a.double()+b.double()[...,self.reflection_action_index]*self.reflection_action_sign)

    def get_distribution(self,obs):
        # Pair the noise scale as well; sign reversal does not change variance.
        paired=.5*(self.log_std+self.log_std[self.reflection_action_index])
        return self.action_dist.proba_distribution(self.mean_action(obs),paired)

    def predict_values(self,obs):
        features=self.extract_features(self._normalize(obs),self.vf_features_extractor)
        return self.value_net(self.mlp_extractor.forward_critic(features))

    def forward(self,obs,deterministic=False):
        dist=self.get_distribution(obs);actions=dist.get_actions(deterministic=deterministic)
        return actions.reshape((-1,*self.action_space.shape)),self.predict_values(obs),dist.log_prob(actions)

    def evaluate_actions(self,obs,actions):
        dist=self.get_distribution(obs)
        return self.predict_values(obs),dist.log_prob(actions),dist.entropy()

    def _get_constructor_parameters(self):
        data=super()._get_constructor_parameters();data['reflection_config']=self.reflection_config
        return data
