import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim/rl'))
try:
    from symmetric_rollout import RawPolicyAdapter
except ImportError:
    RawPolicyAdapter=None

@unittest.skipIf(RawPolicyAdapter is None,'simulator dependencies unavailable')
class RawAdapterTest(unittest.TestCase):
    def env(self):
        class Env:
            active=['left_hip_roll_joint','right_hip_roll_joint','tail_yaw_joint','tail_pitch_joint']
            policy_idx=[0,1,2,3];q0=np.zeros(4);obs_vel=True
            command=np.array([4.,0.,1.])
            raw=np.arange(26,dtype=np.float32)/10
            def _obs(self):return self.raw
        return Env()

    def test_scale_changes_only_internal_raw_yaw_and_preserves_external_state(self):
        env=self.env();before=env.raw.copy();cmd=env.command.copy()
        adapter=RawPolicyAdapter(None,None,env,6.)
        with patch('symmetric_rollout.symmetric_action',return_value=np.zeros(4)) as predict:
            action,_=adapter.predict(np.full((1,26),999.))
        expected=before.copy();expected[8]*=6
        np.testing.assert_array_equal(predict.call_args.args[-1],expected)
        np.testing.assert_array_equal(env.raw,before)
        np.testing.assert_array_equal(env.command,cmd)
        self.assertEqual(action.shape,(1,4))
        with self.assertRaises(ValueError):adapter.predict(None,deterministic=False)

    def test_invalid_scale_rejected(self):
        for scale in [0,-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):RawPolicyAdapter(None,None,self.env(),scale)

if __name__=='__main__':unittest.main()
