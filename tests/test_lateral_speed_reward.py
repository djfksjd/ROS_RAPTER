"""Lateral tracking reward cannot change physics or penalize a requested side velocity."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim/rl'))
try:
 import mujoco
 import numpy as np
 from run_env import RunEnv,CONTROL_DT
except ImportError:mujoco=None
@unittest.skipIf(mujoco is None,'simulator stack unavailable')
class LateralRewardTest(unittest.TestCase):
 def test_default_zero_and_penalty_physics_parity_for_signed_commands(self):
  envs=[RunEnv(randomize=False,seed=7,weights=w) for w in [None,{'lateral_err':0.},{'lateral_err':-2.}]]
  try:
   for cy in [-.3,.3]:
    for e in envs:e.reset(seed=19);e.command=np.array([4.,cy,.5]);e.resample_steps=0
    for _ in range(8):
     r=[e.step(np.zeros(e.action_space.shape)) for e in envs]
     self.assertNotIn('lateral_err',r[0][4]['terms']);self.assertEqual(r[0][1],r[1][1])
     for e in envs[1:]:
      np.testing.assert_array_equal(e.data.qpos,envs[0].data.qpos);np.testing.assert_array_equal(e.data.qvel,envs[0].data.qvel)
     err=abs(cy-r[2][4]['v_body'][1]);self.assertAlmostEqual(r[2][1]-r[0][1],-2.*CONTROL_DT*err,places=13)
     self.assertAlmostEqual(r[2][4]['terms']['lateral_err'],err,places=14)
  finally:
   for e in envs:e.close()
if __name__=='__main__':unittest.main()
