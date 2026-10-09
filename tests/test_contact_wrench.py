"""Physical sign/frame calibration; external forces must balance at rest."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim'/'rl'))
try:
    import mujoco
    import numpy as np
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, 'mujoco unavailable')
class ContactWrenchTest(unittest.TestCase):
    def test_force_and_yaw_moment_balance(self):
        from contact_wrench import contact_wrench
        m = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".001"/><default><geom friction="1 .2 .01" condim="6"/></default><worldbody><geom type="plane" size="0 0 .1"/><body name="box" pos="0 0 .1"><freejoint/><geom type="box" size=".1 .1 .1" mass="2"/></body></worldbody></mujoco>')
        d = mujoco.MjData(m); root = m.body('box').id
        for _ in range(3000):
            d.xfrc_applied[root] = [5, 2, 0, 0, 0, .2]
            mujoco.mj_step(m, d)
        f, tau, contacts = contact_wrench(m, d, root)
        self.assertTrue(contacts)
        np.testing.assert_allclose(f, [-5, -2, 19.62], atol=.01)
        np.testing.assert_allclose(tau, [0, 0, -.2], atol=.01)

    def test_opposite_contact_participant_sign(self):
        from contact_wrench import contact_wrench
        m = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".001"/><worldbody><geom type="plane" size="0 0 .1"/><body name="lower" pos="0 0 .1"><freejoint/><geom type="box" size=".15 .15 .1" mass="2"/></body><body name="upper" pos="0 0 .3"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>')
        d = mujoco.MjData(m)
        for _ in range(3000):
            mujoco.mj_step(m, d)
        for name, weight in [('lower', 19.62), ('upper', 9.81)]:
            f, tau, _ = contact_wrench(m, d, m.body(name).id)
            np.testing.assert_allclose(f, [0, 0, weight], atol=.01)
            np.testing.assert_allclose(tau, [0, 0, 0], atol=.01)


if __name__ == '__main__':
    unittest.main()
