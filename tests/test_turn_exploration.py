"""Targeted action noise preserves propulsion axes and bilateral noise symmetry."""
import sys,unittest,math
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim'/'rl'))
try:
 import torch
 from train_symmetric import set_turn_exploration
except ImportError:torch=None

@unittest.skipIf(torch is None,'sim/torch stack unavailable')
class TurnExplorationTest(unittest.TestCase):
 def test_only_turn_axes_change_and_pair_variances_match(self):
  names=['left_hip_roll_joint','left_hip_pitch_joint','left_knee_pitch_joint','right_hip_roll_joint','right_hip_pitch_joint','right_knee_pitch_joint','tail_yaw_joint','tail_pitch_joint']
  env=SimpleNamespace(active=names,policy_idx=list(range(8)))
  original=torch.tensor([-.9,-1.1,-1.2,-1.3,-1.4,-1.5,-1.6,-1.7])
  policy=SimpleNamespace(log_std=torch.nn.Parameter(original.clone()))
  self.assertEqual(set_turn_exploration(policy,env,.25),[0,3,6])
  torch.testing.assert_close(policy.log_std[[1,2,4,5,7]],original[[1,2,4,5,7]],rtol=0,atol=0)
  torch.testing.assert_close(policy.log_std[[0,3,6]].exp(),torch.full((3,),.25),rtol=0,atol=0)
  self.assertTrue(policy.log_std.requires_grad)

 def test_invalid_noise_or_action_topology_is_rejected(self):
  policy=SimpleNamespace(log_std=torch.nn.Parameter(torch.zeros(1)))
  env=SimpleNamespace(active=['tail_yaw_joint'],policy_idx=[0])
  for value in [0.,-1.,float('inf'),float('nan'),.25]:
   with self.assertRaises(ValueError):set_turn_exploration(policy,env,value)
