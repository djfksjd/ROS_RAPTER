"""Mode-specific support reward leaves dynamics and moving-mode reward frozen."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim'/'rl'))
try:
 import mujoco
 import numpy as np
 from run_env import RunEnv,CONTROL_DT
except ImportError:
 mujoco=None
@unittest.skipIf(mujoco is None,'simulator stack unavailable')
class StandSupportRewardTest(unittest.TestCase):
 def test_reward_difference_and_physics_for_zero_and_moving_commands(self):
  envs=[RunEnv(randomize=False,seed=9,weights=w) for w in (None,{'stand_support':0.},{'stand_support':-5.})]
  observed_nonzero=False
  try:
   for command in ([0.,0.,0.],[4.,0.,0.],[0.,.2,0.],[0.,0.,1.],[.01,0.,0.]):
    for e in envs:
     e.reset(seed=17);e.command=np.array(command);e.resample_steps=0
     # Start above the ground so the integration check exercises real unloaded feet.
     e.data.qpos[2]+=.1
     mujoco.mj_forward(e.model,e.data)
    for _ in range(20):
     results=[e.step(np.zeros(e.action_space.shape)) for e in envs]
     self.assertNotIn('stand_support',results[0][4]['terms'])
     self.assertEqual(results[0][1],results[1][1])
     for e in envs[1:]:
      np.testing.assert_array_equal(e.data.qpos,envs[0].data.qpos)
      np.testing.assert_array_equal(e.data.qvel,envs[0].data.qvel)
      np.testing.assert_array_equal(e._obs(),envs[0]._obs())
     info=results[2][4];support=info['terms']['stand_support']
     if command==[0.,0.,0.]:
      self.assertEqual(support,sum(not v for v in info['loaded'].values())/2)
      observed_nonzero |= support>0
     else:self.assertEqual(support,0.);self.assertEqual(results[0][1],results[2][1])
     self.assertAlmostEqual(results[2][1]-results[0][1],-5.*CONTROL_DT*support,places=13)
     if any(r[2] for r in results):break
   self.assertTrue(observed_nonzero,'must exercise a real missing-support state')
  finally:
   for e in envs:e.close()
