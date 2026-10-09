"""Torque use must be compared to the bound for its own sign."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim'/'rl'))
try:
 import mujoco
 import numpy as np
 from actuator_turn_probe import utilization
except ImportError:mujoco=None

@unittest.skipIf(mujoco is None,'simulator stack unavailable')
class ActuatorTurnProbeTest(unittest.TestCase):
 def test_sign_specific_bounds_and_zero_capacity(self):
  r=utilization(np.array([9.,-4.,0.]),np.array([-5.,-5.,0.]),np.array([10.,20.,0.]))
  np.testing.assert_array_equal(r,[.9,.8,0.]);self.assertTrue(np.isfinite(r).all())
 def test_mirrored_torque_and_bounds_preserve_utilization(self):
  torque=np.array([3.,-4.,2.]);low=np.array([-5.,-7.,-6.]);high=np.array([8.,10.,9.])
  np.testing.assert_array_equal(utilization(torque,low,high),utilization(-torque,-high,-low))
 def test_reset_dependent_actuators_and_readonly_trajectory(self):
  import json,tempfile
  from unittest.mock import patch
  from actuator_turn_probe import probe
  args=dict(mass=5.,springs='c',tail='active',level=0.,cmd=[4.,0.,1.],episode_s=15.,weights='{}',zero_cmd=.2,top_cmd=.2,disturb_items=[],no_obs_vel=False,no_init_speed=True,ankle_clutch=False)
  class ZeroPolicy:
   def __init__(self,env):self.env=env
   def predict(self,obs,deterministic=True):return np.zeros((1,*self.env.action_space.shape)),None
  class IdentityNorm:
   def __init__(self,env):self.env=env
   def reset(self):return self.env.reset()[0][None]
   def normalize_obs(self,obs):return obs
   def close(self):self.env.close()
  with tempfile.TemporaryDirectory() as folder:
   (Path(folder)/'args.json').write_text(json.dumps(args))
   with patch('actuator_turn_probe.load',side_effect=lambda path,env:(ZeroPolicy(env),IdentityNorm(env))):
    plain=probe(folder,1.,41,False);measured=probe(folder,1.,41,True)
   self.assertGreater(len(measured['controls']),0)
   self.assertEqual(plain['controls'],measured['controls'])
   self.assertEqual(plain['fell'],measured['fell'])
