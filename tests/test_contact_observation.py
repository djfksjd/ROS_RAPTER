"""Optional loaded flags preserve physics, mirror contact identity and transfer Adam state."""
import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim'/'rl'))
try:
 import mujoco,numpy as np,torch,gymnasium as gym
 from stable_baselines3 import PPO
 from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
 from run_env import RunEnv
 from mirror_probe import ObservationReflection
 from symmetric_policy import SymmetricPolicy,configuration
 from train_contact_observation import transfer_policy,INPUT_WEIGHTS
except ImportError:mujoco=None

@unittest.skipIf(mujoco is None,'simulator/torch unavailable')
class ContactObservationTest(unittest.TestCase):
 def test_flags_append_without_changing_physics_or_old_features(self):
  envs=[RunEnv(arch8=True,randomize=False,seed=3,obs_contact=flag) for flag in [False,True]]
  try:
   obs=[e.reset(seed=3)[0] for e in envs]
   self.assertEqual([len(x) for x in obs],[46,48]);self.assertEqual([e.action_space.shape for e in envs],[(8,),(8,)])
   for _ in range(12):
    np.testing.assert_array_equal(obs[0],obs[1][:-2])
    self.assertEqual(obs[1][-2:].tolist(),[float(envs[1].loaded[s]) for s in ['left','right']])
    result=[e.step(np.zeros(8)) for e in envs];obs=[r[0] for r in result]
    self.assertEqual(result[0][1:4],result[1][1:4])
    np.testing.assert_array_equal(envs[0].data.qpos,envs[1].data.qpos)
    np.testing.assert_array_equal(envs[0].data.qvel,envs[1].data.qvel)
  finally:
   for e in envs:e.close()

 def test_reflection_swaps_flags_without_flipping_their_sign(self):
  e=RunEnv(arch8=True,randomize=False,obs_contact=True)
  try:
   raw=e.reset()[0];raw[-2:]=[0.,1.]
   r=ObservationReflection(e.active,e.policy_idx,e.q0,e.obs_vel,True);mirror=r.observation(raw)
   np.testing.assert_array_equal(mirror[-2:],[1.,0.]);np.testing.assert_array_equal(mirror[-4:-2],-raw[-4:-2])
   np.testing.assert_allclose(r.observation(mirror),raw,rtol=0,atol=1e-7)
  finally:e.close()

 def test_policy_adam_transfer_gradients_and_saved_restore(self):
  envs=[RunEnv(arch8=True,randomize=False,obs_contact=flag) for flag in [False,True]]
  norms=[];models=[]
  try:
   for e in envs:
    norm=VecNormalize(DummyVecEnv([lambda e=e:e]),norm_obs=False,norm_reward=False);norms.append(norm)
    size=e.observation_space.shape[0];stats=SimpleNamespace(obs_rms=SimpleNamespace(mean=np.zeros(size),var=np.ones(size)),epsilon=0.,clip_obs=10.)
    models.append(PPO(SymmetricPolicy,norm,n_steps=8,batch_size=8,n_epochs=1,seed=7,policy_kwargs={'reflection_config':configuration(e,stats,6.4)}))
   old,new=models;raw=torch.tensor(np.random.default_rng(91).normal(size=(12,46)),dtype=torch.float32)
   _,lp,entropy=old.policy.evaluate_actions(raw,torch.zeros((12,8)))
   loss=old.policy.predict_values(raw).square().mean()-lp.mean()-.01*entropy.mean();loss.backward();old.policy.optimizer.step();old.policy.optimizer.zero_grad()
   transfer_policy(old.policy,new.policy,2)
   expanded=torch.cat([raw,torch.tensor([[0.,1.]]).repeat(12,1)],dim=1)
   with torch.no_grad():
    torch.testing.assert_close(old.policy.mean_action(raw),new.policy.mean_action(expanded),rtol=0,atol=1e-7)
    torch.testing.assert_close(old.policy.predict_values(raw),new.policy.predict_values(expanded),rtol=0,atol=1e-6)
   params_old=dict(old.policy.named_parameters());params_new=dict(new.policy.named_parameters())
   for key in INPUT_WEIGHTS:
    param=params_new[key];self.assertEqual(torch.count_nonzero(param[:,-2:]).item(),0)
    for field in ['exp_avg','exp_avg_sq']:
     state=new.policy.optimizer.state[param][field];self.assertEqual(state.shape,param.shape)
     torch.testing.assert_close(state[:,:-2],old.policy.optimizer.state[params_old[key]][field],rtol=0,atol=0)
     self.assertEqual(torch.count_nonzero(state[:,-2:]).item(),0)
   _,lp,entropy=new.policy.evaluate_actions(expanded,torch.ones((12,8))*.2)
   (new.policy.predict_values(expanded).square().mean()-lp.mean()).backward()
   self.assertGreater(float(params_new['mlp_extractor.policy_net.0.weight'].grad[:,-2:].abs().sum()),0.)
   new.policy.optimizer.step()  # new columns and transferred Adam tensors work together
   with tempfile.TemporaryDirectory() as d:
    path=Path(d);new.save(path/'model');norms[1].save(path/'norm.pkl');loaded=PPO.load(path/'model')
    restored=VecNormalize.load(str(path/'norm.pkl'),DummyVecEnv([lambda:RunEnv(arch8=True,randomize=False,obs_contact=True)]))
    try:
     self.assertEqual(restored.observation_space.shape,(48,))
     np.testing.assert_array_equal(new.predict(expanded.numpy(),deterministic=True)[0],loaded.predict(expanded.numpy(),deterministic=True)[0])
    finally:restored.close()
  finally:
   for n in norms:n.close()
