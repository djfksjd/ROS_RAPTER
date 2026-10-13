"""Left-right symmetric actor-critic for RunEnv (opt-in, train_run.py --symmetric).

The action mean is 0.5*(mu(o) + Ma mu(Mo o)) and the value 0.5*(V(o) + V(Mo o)), where Mo / Ma are the sagittal
reflections of mirror_probe.ObservationReflection (legs exchanged, roll/yaw signs flipped, stride clock shifted by half
a cycle). The policy is therefore exactly equivariant: a mirrored situation gets the mirrored action, so neither leg can
be systematically loaded more than the other. The reflection acts on raw observations, so the observation normalisation
is frozen (FrozenRMS) and its mean / variance are stored with the policy.
"""
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.running_mean_std import RunningMeanStd


class FrozenRMS(RunningMeanStd):
    """Observation statistics that no longer update (the symmetric policy stores them)."""

    def update(self, arr):
        return None


def freeze_obs(venv):
    f = FrozenRMS(shape=venv.obs_rms.mean.shape)
    f.mean, f.var, f.count = venv.obs_rms.mean.copy(), venv.obs_rms.var.copy(), venv.obs_rms.count
    venv.obs_rms = f
    return venv


def reflection_constants(reflection, venv):
    """Signed permutation + offset form of the raw-observation reflection, and the frozen normalisation."""
    n = reflection.size
    c = reflection.observation(np.zeros(n))
    J = np.array([reflection.observation(np.eye(n)[i])-c for i in range(n)]).T  # y = J x + c
    idx = np.abs(J).argmax(1)
    sign = J[np.arange(n), idx]
    assert np.allclose(np.abs(sign), 1.) and np.allclose(np.abs(J).sum(1), 1.)
    return dict(idx=idx.tolist(), sign=sign.tolist(), off=c.tolist(), a_idx=reflection.action_index.tolist(), a_sign=reflection.action_sign.tolist(),
                mean=venv.obs_rms.mean.tolist(), std=np.sqrt(venv.obs_rms.var+venv.epsilon).tolist(), clip=float(venv.clip_obs))


class SymPolicy(ActorCriticPolicy):
    def __init__(self, *args, sym=None, **kwargs):
        super().__init__(*args, **kwargs)
        t = lambda k, dt=th.float32: th.as_tensor(sym[k], dtype=dt)
        self.sym_kwargs = sym
        for k in ('sign', 'off', 'a_sign', 'mean', 'std'):
            self.register_buffer('sym_'+k, t(k), persistent=False)
        self.register_buffer('sym_idx', t('idx', th.long), persistent=False)
        self.register_buffer('sym_a_idx', t('a_idx', th.long), persistent=False)
        self.sym_clip = float(sym['clip'])

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data['sym'] = self.sym_kwargs
        return data

    def mirror_obs(self, obs):
        raw = obs*self.sym_std+self.sym_mean
        m = raw[..., self.sym_idx]*self.sym_sign+self.sym_off
        return th.clamp((m-self.sym_mean)/self.sym_std, -self.sym_clip, self.sym_clip)

    def _sym(self, obs, actor=True, critic=True):
        both = th.cat([obs, self.mirror_obs(obs)], 0)
        f = self.extract_features(both)
        n = obs.shape[0]
        mean = value = None
        if actor:
            mu = self.action_net(self.mlp_extractor.forward_actor(f))
            mean = .5*(mu[:n]+mu[n:][..., self.sym_a_idx]*self.sym_a_sign)
        if critic:
            v = self.value_net(self.mlp_extractor.forward_critic(f))
            value = .5*(v[:n]+v[n:])
        return mean, value

    def forward(self, obs, deterministic=False):
        mean, values = self._sym(obs)
        dist = self.action_dist.proba_distribution(mean, self.log_std)
        actions = dist.get_actions(deterministic=deterministic)
        return actions.reshape((-1, *self.action_space.shape)), values, dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        mean, values = self._sym(obs)
        dist = self.action_dist.proba_distribution(mean, self.log_std)
        return values, dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        return self.action_dist.proba_distribution(self._sym(obs, critic=False)[0], self.log_std)

    def predict_values(self, obs):
        return self._sym(obs, actor=False)[1]
