import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim/rl'))
from yaw_feedback import YawCommandFeedback, heading_rate

class YawFeedbackTest(unittest.TestCase):
    def test_disabled_and_zero_command_preserve_feedforward(self):
        controller=YawCommandFeedback(scale=6.4)
        for desired in [-1.,-.5,0.,.5,1.]:
            self.assertEqual(controller.update(desired,3.),6.4*desired)
        self.assertEqual(YawCommandFeedback(gain=4).update(0.,-20.),0.)

    def test_correction_sign_filter_and_clamp(self):
        pos=YawCommandFeedback(gain=4);neg=YawCommandFeedback(gain=4)
        self.assertEqual(pos.update(1,0),8.)
        self.assertEqual(neg.update(-1,0),-8.)
        for _ in range(100):a=pos.update(1,1.5);b=neg.update(-1,-1.5)
        self.assertAlmostEqual(a,-b);self.assertLess(a,6.4)
        c=YawCommandFeedback(gain=4);c.update(1,1,dt=.2)
        self.assertAlmostEqual(c.filtered_rate,1-math.exp(-1))

    def test_correction_bound_preserves_feedforward_neighborhood(self):
        c=YawCommandFeedback(gain=4,correction_limit=.4)
        for desired,measured in [(1,0),(.5,-5),(-.5,5),(-1,0)]:
            self.assertLessEqual(abs(c.update(desired,measured)-6.4*desired),.40000000000001)
        for limit in [0,-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):YawCommandFeedback(correction_limit=limit)

    def test_heading_rate_with_banked_and_pitched_body(self):
        import numpy as np
        for roll,pitch,yaw in [(0,0,0),(.5,.3,.7),(-.5,-.4,-.9)]:
            cr,sr=math.cos(roll),math.sin(roll);cp,sp=math.cos(pitch),math.sin(pitch)
            cy,sy=math.cos(yaw),math.sin(yaw)
            R=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])@np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])@np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
            # A pure world yaw rotation must remain the same heading rate at all tilts.
            self.assertAlmostEqual(heading_rate(R,R.T@np.array([0,0,1.2])),1.2)
        with self.assertRaises(ValueError):heading_rate([[0,0,1],[0,1,0],[-1,0,0]],[0,0,1])

    def test_nonfinite_and_invalid_values_rejected(self):
        for kw in [{'gain':-1},{'time_constant':0},{'limit':0},{'scale':float('nan')}]:
            with self.assertRaises(ValueError):YawCommandFeedback(**kw)
        for args in [(float('inf'),0,.02),(0,float('nan'),.02),(0,0,0)]:
            with self.assertRaises(ValueError):YawCommandFeedback().update(*args)

if __name__=='__main__':unittest.main()
