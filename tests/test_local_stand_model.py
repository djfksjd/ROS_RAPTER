"""Reference replay, target-ramp memory, and stricter current limit contracts."""
import sys
import json
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sim/rl'))
try:
    import mujoco
    import numpy as np
    from local_stand_model import LocalStandModel
except ImportError:
    mujoco=None


@unittest.skipIf(mujoco is None,'simulator stack unavailable')
class LocalStandModelTest(unittest.TestCase):
    def setUp(self):
        source=ROOT/'sim/rl/runs/kl005_lr3e5_update64k_20261009'
        records=ROOT/'docs/evidence/148-static-equilibrium/margin-search.jsonl'
        matrix=ROOT/'docs/evidence/150-local-linearization/epsilon_1e-06.npz'
        if not source.exists() or not records.exists() or not matrix.exists():
            self.skipTest('archived run fixtures unavailable')
        candidate=next(x for x in map(json.loads,records.read_text().splitlines()) if x['start_knee']==1.1)
        self.model=LocalStandModel(source,candidate)
        self.baseline=np.load(matrix)['baseline_step']

    def tearDown(self):
        if hasattr(self,'model'):self.model.close()

    def test_reference_replays_exactly_with_archived_map(self):
        x=np.zeros(self.model.nx);u=np.zeros(8)
        np.testing.assert_array_equal(self.model.transition(x,u),self.baseline)
        np.testing.assert_array_equal(self.model.transition(x,u),self.baseline)

    def test_previous_target_changes_physical_next_state(self):
        x=np.zeros(self.model.nx);u=np.zeros(8)
        baseline=self.model.transition(x,u)
        x[2*self.model.nv+1]=.002
        changed=self.model.transition(x,u)
        self.assertGreater(np.linalg.norm(changed[:2*self.model.nv]-baseline[:2*self.model.nv]),1e-6)

    def test_current_limit_never_expands_motor_envelope(self):
        x=np.zeros(self.model.nx);u=np.zeros(8)
        self.model.restore(x);low,high=self.model.env._clamp()
        self.model.set_stand_current_limit(.995)
        limited_low,limited_high=self.model.env._clamp()
        self.assertTrue(np.all(limited_low>=low))
        self.assertTrue(np.all(limited_high<=high))
        np.testing.assert_array_equal(self.model.transition(x,u),self.baseline)
        self.model.set_stand_current_limit(None)
        np.testing.assert_array_equal(self.model.transition(x,u),self.baseline)
        for factor in [0.,-1.,1.01,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):self.model.set_stand_current_limit(factor)

    def test_invalid_inputs_do_not_restore_or_step(self):
        before=self.model.env.data.qpos.copy()
        with self.assertRaises(ValueError):self.model.transition(np.zeros(self.model.nx),np.full(8,np.nan))
        with self.assertRaises(ValueError):self.model.restore(np.zeros(self.model.nx+1))
        np.testing.assert_array_equal(self.model.env.data.qpos,before)
