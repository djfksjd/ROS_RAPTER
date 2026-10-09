"""Sensor-only controller contract, frozen design, and state reset behavior."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sim/rl'))
try:
    import numpy as np
    from stand_observer import StandObserverController
except ImportError:
    np=None


@unittest.skipIf(np is None,'numpy unavailable')
class StandObserverTest(unittest.TestCase):
    def setUp(self):
        path=ROOT/'docs/evidence/153-stand-observer/observer_design.npz'
        if not path.exists():self.skipTest('observer fixture unavailable')
        with np.load(path) as archive:
            self.design={key:archive[key].copy() for key in archive.files}
        self.controller=StandObserverController(self.design)

    def test_reference_hold_and_reset_with_no_environment(self):
        self.assertFalse(hasattr(self.controller,'env'))
        observation=self.design['reference'].copy()
        np.testing.assert_array_equal(self.controller.act(observation),self.design['u_reference'])
        changed=observation.copy();changed[0]+=.003
        self.controller.act(changed)
        self.controller.reset()
        np.testing.assert_array_equal(self.controller.act(observation),self.design['u_reference'])

    def test_design_is_frozen_and_action_respects_physical_bounds(self):
        original_K=self.controller.K.copy()
        self.design['Kc'][:]=1e6
        np.testing.assert_array_equal(self.controller.K,original_K)
        observation=self.design['reference'].copy();observation[:3]+=100.
        action=self.controller.act(observation)
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue(np.all(action>=self.controller.action_lower))
        self.assertTrue(np.all(action<=self.controller.action_upper))

    def test_invalid_observation_and_nonzero_command_preserve_state(self):
        prior=self.controller.prior.copy()
        wrong_command=self.design['reference'].copy();wrong_command[6]=.4
        for observation in [np.zeros(48),np.full(46,np.nan),wrong_command]:
            with self.assertRaises(ValueError):self.controller.act(observation)
            np.testing.assert_array_equal(self.controller.prior,prior)
        bad=dict(self.design);bad['action_upper']=np.full(8,2.)
        with self.assertRaises(ValueError):StandObserverController(bad)
