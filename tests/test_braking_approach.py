import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim/rl'))
try:
 import numpy as np
 from braking_approach import BrakingApproach
except ImportError:
 np=None

@unittest.skipIf(np is None,'numpy unavailable')
class BrakingApproachTest(unittest.TestCase):
 def setUp(self):
  self.indices=np.array([0,1,2,5,6,7,10,11]);kp=np.zeros(12);kp[self.indices]=100.
  self.design=dict(q0=np.zeros(12),kp=kp,scale=np.full(12,.5),policy_indices=self.indices,spring_stiffness=np.array([0,0,10,0,0,10,0,0]),spring_rest=np.zeros(8),reference_action=np.zeros(8),lower=np.full(8,-1.),upper=np.ones(8))
  self.obs=np.zeros(46);self.obs[12:24]=.02
 def test_braking_sign_and_unactuated_zero_gains(self):
  p=BrakingApproach(self.obs,**self.design)
  self.assertFalse(hasattr(p,'env'))
  np.testing.assert_allclose(p.initial,[.04,.04,.044,.04,.04,.044,.04,.04])
  self.assertGreater(p.initial[2],.04)
 def test_independent_and_symmetric_modes(self):
  self.obs[12+self.indices[2]]=.04
  independent=BrakingApproach(self.obs,**self.design);symmetric=BrakingApproach(self.obs,**self.design,symmetric_pairs=[(2,5)])
  self.assertNotEqual(independent.initial[2],independent.initial[5])
  self.assertEqual(symmetric.initial[2],symmetric.initial[5])
  np.testing.assert_allclose(symmetric.initial[[2,5]],np.mean(independent.initial[[2,5]]))
 def test_action_limits_and_exact_final_reference(self):
  self.obs[12:24]=10.;p=BrakingApproach(self.obs,**self.design)
  for t in [0,.1,.2,1.]:
   action=p.act(t);self.assertTrue(np.all(action<=1) and np.all(action>=-1))
  np.testing.assert_array_equal(p.act(.2),self.design['reference_action'])
  p.act(.2)[:]=1;np.testing.assert_array_equal(p.act(1),self.design['reference_action'])
 def test_invalid_command_design_and_time_rejected(self):
  bad=self.obs.copy();bad[6]=1
  with self.assertRaises(ValueError):BrakingApproach(bad,**self.design)
  kp=self.design['kp'].copy();kp[0]=0
  with self.assertRaises(ValueError):BrakingApproach(self.obs,**{**self.design,'kp':kp})
  indices=self.indices.copy();indices[1]=indices[0]
  with self.assertRaises(ValueError):BrakingApproach(self.obs,**{**self.design,'policy_indices':indices})
  p=BrakingApproach(self.obs,**self.design)
  for t in [-1,np.nan,np.inf]:
   with self.assertRaises(ValueError):p.act(t)
