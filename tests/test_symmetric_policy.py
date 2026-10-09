import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim/rl'))
try:
    import torch
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    from symmetric_policy import SymmetricPolicy,configuration
except ImportError:SymmetricPolicy=None

@unittest.skipIf(SymmetricPolicy is None,'simulator dependencies unavailable')
class SymmetricPolicyTest(unittest.TestCase):
    def setup(self):
        class Env(gym.Env):
            active=['left_hip_roll_joint','right_hip_roll_joint','tail_yaw_joint','tail_pitch_joint']
            policy_idx=[0,1,2,3];q0=np.array([.1,-.1,0,.2]);obs_vel=True
            observation_space=gym.spaces.Box(-np.inf,np.inf,(26,),dtype=np.float32)
            action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32)
            def reset(self,seed=None,options=None):super().reset(seed=seed);return np.zeros(26,np.float32),{}
            def step(self,a):return np.zeros(26,np.float32),0.,False,True,{}
        env=Env();norm=VecNormalize(DummyVecEnv([lambda:env]));norm.obs_rms.mean=np.linspace(-.3,.3,26);norm.obs_rms.var=np.linspace(.5,1.5,26)
        config=configuration(env,norm,6.4);norm.norm_obs=False
        model=PPO(SymmetricPolicy,norm,policy_kwargs={'reflection_config':config,'net_arch':{'pi':[16],'vf':[16]}},n_steps=8,batch_size=8,n_epochs=1,seed=7)
        return model,norm

    def test_mean_and_noise_equivariance_and_probability_consistency(self):
        model,norm=self.setup();obs=torch.tensor(np.random.default_rng(3).normal(size=(9,26)),dtype=torch.float32)
        p=model.policy
        with torch.no_grad():
            a=p.mean_action(obs);b=p.mean_action(p.mirror(obs).float())
            torch.testing.assert_close(a,b[:,p.reflection_action_index]*p.reflection_action_sign,atol=2e-7,rtol=1e-5)
            actions,values,logp=p(obs)
            values2,logp2,entropy=p.evaluate_actions(obs,actions)
            torch.testing.assert_close(values,values2);torch.testing.assert_close(logp,logp2)
            self.assertTrue(torch.isfinite(entropy).all())
        norm.close()

    def test_actor_gradients_and_save_reload(self):
        model,norm=self.setup();p=model.policy;obs=torch.ones((4,26))*.1
        values,logp,entropy=p.evaluate_actions(obs,torch.ones((4,4))*.2)
        (values.sum()-logp.sum()-.001*entropy.sum()).backward()
        self.assertGreater(float(p.action_net.weight.grad.abs().sum()),0.)
        self.assertTrue(torch.isfinite(p.action_net.weight.grad).all())
        before,_=model.predict(obs.numpy(),deterministic=True)
        with tempfile.TemporaryDirectory() as d:
            model.save(Path(d)/'model');norm.save(Path(d)/'vecnorm.pkl')
            loaded=PPO.load(Path(d)/'model');after,_=loaded.predict(obs.numpy(),deterministic=True)
            np.testing.assert_array_equal(before,after)
            self.assertEqual(loaded.policy.reflection_config,p.reflection_config)
        norm.close()

if __name__=='__main__':unittest.main()
