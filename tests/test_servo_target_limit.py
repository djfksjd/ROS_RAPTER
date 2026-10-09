import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim/rl'))
try:
 import numpy as np
 from servo_target_limit import limit_servo_action
except ImportError:
 np=None

@unittest.skipIf(np is None,'numpy unavailable')
class ServoTargetLimitTest(unittest.TestCase):
 def setUp(self):
  self.design=dict(q=np.full(8,.2),qd=np.ones(8),q0=np.zeros(8),scale=np.ones(8),kp=np.full(8,100.),kd=np.full(8,2.),torque_lower=np.full(8,-10.),torque_upper=np.full(8,20.),action_lower=np.full(8,-1.),action_upper=np.ones(8))
 def test_requested_pd_torque_inside_asymmetric_motor_interval(self):
  d=self.design;action,empty=limit_servo_action(np.linspace(-1,1,8),**d)
  torque=d['kp']*(d['q0']+d['scale']*action-d['q'])-d['kd']*d['qd']
  self.assertFalse(empty.any());self.assertTrue(np.all(torque>=-10.-1e-10));self.assertTrue(np.all(torque<=20.+1e-10))
  self.assertAlmostEqual(action[0],.12);self.assertAlmostEqual(action[-1],.42)
 def test_feasible_action_is_unchanged(self):
  original=np.full(8,.3);action,empty=limit_servo_action(original,**self.design)
  np.testing.assert_array_equal(action,original);self.assertFalse(empty.any())
 def test_empty_intersection_reported_and_nearest_physical_endpoint(self):
  action,empty=limit_servo_action(np.zeros(8),**{**self.design,'q':np.full(8,5.)})
  self.assertTrue(empty.all());np.testing.assert_array_equal(action,np.ones(8))
 def test_invalid_vectors_ranges_and_overflow_rejected(self):
  for changes in [{'kp':np.zeros(8)},{'torque_upper':np.full(8,-20.)},{'q':np.zeros(9)},{'qd':np.full(8,np.nan)},{'kd':np.full(8,1e308),'qd':np.full(8,1e308)}]:
   with self.assertRaises(ValueError):limit_servo_action(np.zeros(8),**{**self.design,**changes})
