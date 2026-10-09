import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim/rl'))
try:from roll_range_probe import hip_tracking_summary
except ImportError:hip_tracking_summary=None

@unittest.skipIf(hip_tracking_summary is None,'simulator dependencies unavailable')
class RangeSummaryTest(unittest.TestCase):
    def row(self,t):return dict(t=t,action=[-1,1],target=[-.3,.3],actual=[-.2,.2])
    def test_no_turn_samples_are_missing_measurement_not_zero_saturation(self):
        result=hip_tracking_summary([self.row(.2)],1.)
        self.assertEqual(result['measured_samples'],0)
        self.assertIsNone(result['action_saturation_fraction'])
        self.assertIsNone(result['hip_tracking_rms_rad'])
        json.dumps(result,allow_nan=False)

    def test_exact_turn_interval_and_known_tracking_error(self):
        result=hip_tracking_summary([self.row(3.98),self.row(4.),self.row(6.)],1.)
        self.assertEqual(result['measured_samples'],1)
        self.assertEqual(result['action_saturation_fraction'],[1.,1.])
        self.assertAlmostEqual(result['hip_tracking_rms_rad'][0],.1)
        self.assertAlmostEqual(result['hip_tracking_rms_rad'][1],.1)

if __name__=='__main__':unittest.main()
