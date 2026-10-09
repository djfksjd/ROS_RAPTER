"""Interior averaged means can still have two locally clipped branches."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim'/'rl'))
try:
 import torch
 import numpy as np
 from actor_clip_probe import clipping_flags
except ImportError:torch=None

@unittest.skipIf(torch is None,'simulator/torch stack unavailable')
class ActorClipProbeTest(unittest.TestCase):
 def test_opposite_clipping_and_output_derivatives(self):
  a=torch.tensor([2.,.3,2.],requires_grad=True);b=torch.tensor([-3.,.2,3.],requires_grad=True)
  mean=.5*(a.clamp(-1,1)+b.clamp(-1,1));mean.sum().backward()
  both,cancel=clipping_flags(a.detach().numpy(),b.detach().numpy())
  np.testing.assert_array_equal(both,[True,False,True]);np.testing.assert_array_equal(cancel,[True,False,False])
  self.assertEqual(mean[0].item(),0.)
  torch.testing.assert_close(a.grad,torch.tensor([0.,.5,0.]),rtol=0,atol=0)
  torch.testing.assert_close(b.grad,torch.tensor([0.,.5,0.]),rtol=0,atol=0)
