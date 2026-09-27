import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/raptor_control/scripts'))
from motion_guard import TiltGuard


class TiltTests(unittest.TestCase):
    def test_yaw_does_not_count_as_gravity_tilt(self):
        g=TiltGuard();g.observe(0,0,1,0);g.check()
        self.assertEqual(g.tilt,0)

    def test_failure_latches(self):
        g=TiltGuard();g.observe(math.sin(.3/2),0,0,math.cos(.3/2))
        with self.assertRaisesRegex(RuntimeError,'exceeded'):g.check()
        g.observe(0,0,0,1)
        with self.assertRaisesRegex(RuntimeError,'exceeded'):g.check()

    def test_invalid_and_missing_and_stale(self):
        now=[0.];g=TiltGuard(clock=lambda:now[0])
        with self.assertRaises(RuntimeError):g.check()
        g.observe(0,0,0,1);g.check();now[0]=2
        with self.assertRaises(RuntimeError):g.check()
        for q in [(0,0,0,0),(float('nan'),0,0,1)]:
            g=TiltGuard();g.observe(*q)
            with self.assertRaises(RuntimeError):g.check()


if __name__=='__main__':unittest.main()
