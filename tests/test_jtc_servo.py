"""JTCLikeServo: sampled goals reached linearly over the horizon; STOP holds."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sim'))
try:
    import mujoco
except ImportError:  # .venv-ai does not install MuJoCo; use .venv-sim
    mujoco = None


@unittest.skipIf(mujoco is None, 'mujoco not installed')
class JTCLikeServoTest(unittest.TestCase):
    def setUp(self):
        from raptor_servo import JTCLikeServo
        self.model = mujoco.MjModel.from_xml_path(str(ROOT/'sim/raptor.xml'))
        self.data = mujoco.MjData(self.model)
        self.servo = JTCLikeServo(self.model, sample=.02, horizon=.02)
        self.servo.step(self.data)
        self.joint = 'left_hip_roll_joint'
        self.index = 0

    def run_for(self, seconds, target=None):
        for _ in range(round(seconds/self.model.opt.timestep)):
            if target is not None:
                self.servo.set_target({self.joint: target})
            self.servo.step(self.data)

    def test_goal_reached_after_horizon_not_before(self):
        start = self.servo.command[self.index]
        self.servo.set_target({self.joint: start+.1})
        self.run_for(.01)
        middle = self.servo.command[self.index]-start
        self.run_for(.02)
        self.assertGreater(middle, 0)
        self.assertLess(middle, .1)
        self.assertAlmostEqual(self.servo.command[self.index]-start, .1, places=9)

    def test_targets_are_sampled_not_continuous(self):
        start = self.servo.command[self.index]
        self.servo.set_target({self.joint: start+.1})
        self.servo.set_target({self.joint: start+.3})  # same sample period: ignored
        self.run_for(.03)
        self.assertAlmostEqual(self.servo.command[self.index]-start, .1, places=9)

    def test_stop_holds_read_positions(self):
        self.run_for(.05, target=.2)
        self.servo.stop()
        held = self.servo.command.copy()
        self.run_for(.05)
        self.assertTrue((self.servo.command == held).all())

    def test_delay_holds_the_goal_back(self):
        from raptor_servo import JTCLikeServo
        servo = JTCLikeServo(self.model, sample=.02, horizon=.02, delay=.04)
        servo.step(self.data)
        start = servo.command[self.index]
        servo.set_target({self.joint: start+.1})
        for _ in range(round(.035/self.model.opt.timestep)):
            servo.step(self.data)
        self.assertAlmostEqual(servo.command[self.index], start)  # nothing before 40 ms
        for _ in range(round(.04/self.model.opt.timestep)):  # past the first 10 ms controller tick after the ramp
            servo.step(self.data)
        self.assertAlmostEqual(servo.command[self.index]-start, .1, places=9)  # 40 ms delay + 20 ms ramp


if __name__ == '__main__':
    unittest.main()
