"""Optional speed penalty changes reward, never simulator trajectories."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim'/'rl'))
try:
 import mujoco
 import numpy as np
 from run_env import RunEnv,CONTROL_DT
except ImportError:
 mujoco=None

@unittest.skipIf(mujoco is None,'simulator stack unavailable')
class DenseSpeedRewardTest(unittest.TestCase):
 def test_optional_penalty_preserves_physics_and_default_reward(self):
  envs=[RunEnv(randomize=False,seed=9,weights=w) for w in [None,{'lin_err':0.},{'lin_err':-2.}]]
  try:
   for e in envs:e.reset(seed=17);e.command=np.zeros(3);e.resample_steps=0
   for _ in range(8):
    result=[e.step(np.zeros(e.action_space.shape)) for e in envs]
    self.assertNotIn('lin_err',result[0][4]['terms'])
    self.assertEqual(result[0][1],result[1][1])
    for e in envs[1:]:
     np.testing.assert_array_equal(e.data.qpos,envs[0].data.qpos)
     np.testing.assert_array_equal(e.data.qvel,envs[0].data.qvel)
    info=result[2][4];err=float(np.linalg.norm(info['command'][:2]-info['v_body'][:2]))
    self.assertEqual(info['terms']['lin_err'],err)
    self.assertAlmostEqual(result[2][1]-result[0][1],-2.*CONTROL_DT*err,places=13)
  finally:
   for e in envs:e.close()

 def test_large_braking_errors_remain_distinct(self):
  # The actual post-step velocity is used, rather than an assumed initial velocity.
  env=RunEnv(randomize=False,seed=7,weights={'lin_err':-2.})
  errors=[]
  try:
   for speed in [2.,4.]:
    env.reset(seed=17);env.command=np.zeros(3);env.resample_steps=0
    env.data.qvel[:2]=[speed,.25]
    mujoco.mj_forward(env.model,env.data)
    _,_,_,_,info=env.step(np.zeros(env.action_space.shape))
    errors.append(info['terms']['lin_err'])
    self.assertAlmostEqual(errors[-1],np.linalg.norm(info['v_body'][:2]),places=14)
    self.assertLess(info['terms']['track_lin'],1e-10)
   self.assertGreater(errors[1]-errors[0],1.)
  finally:env.close()
