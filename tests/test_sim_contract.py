"""MuJoCo model must keep the Raptor robot contract of the generated URDF."""
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'sim'), str(ROOT/'src/raptor_control/scripts')]
try:
    import mujoco
except ImportError:  # .venv-ai does not install MuJoCo; use .venv-sim
    mujoco = None


@unittest.skipIf(mujoco is None, 'mujoco not installed')
class SimContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lateral_feasibility import Model
        from raptor_servo import ACTIVE
        cls.active = ACTIVE
        cls.urdf_text = (ROOT/'sim/raptor_passive_toes.urdf').read_text()
        cls.urdf = ET.fromstring(cls.urdf_text)
        cls.fk = Model(cls.urdf_text)
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT/'sim/raptor.xml'))

    def test_exactly_ten_actuators_on_active_joints(self):
        names = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.assertEqual(names, self.active)
        self.assertEqual(len(names), 10)
        joints = {j.get('name'): j for j in self.urdf.findall('joint')}
        for i, name in enumerate(names):
            limit = joints[name].find('limit')
            self.assertEqual(self.model.actuator_trnid[i, 0], self.model.joint(name).id)
            np.testing.assert_allclose(self.model.actuator_forcerange[i], [-float(limit.get('effort')), float(limit.get('effort'))])
            np.testing.assert_allclose(self.model.actuator_ctrlrange[i], [-float(limit.get('velocity')), float(limit.get('velocity'))])

    def test_joints_axes_limits_and_passive_springs(self):
        revolute = {j.get('name'): j for j in self.urdf.findall('joint') if j.get('type') == 'revolute'}
        mj = {self.model.joint(i).name: i for i in range(self.model.njnt)
              if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE}
        self.assertEqual(set(mj), set(revolute))
        self.assertEqual(len(set(mj)-set(self.active)), 12)
        for name, joint in revolute.items():
            i = mj[name]
            np.testing.assert_allclose(self.model.jnt_axis[i], list(map(float, joint.find('axis').get('xyz').split())))
            np.testing.assert_allclose(self.model.jnt_range[i], [float(joint.find('limit').get(k)) for k in ('lower', 'upper')])
            if name not in self.active:
                self.assertGreater(self.model.jnt_stiffness[i], 0)

    def test_mass_and_zero_pose_com(self):
        self.assertAlmostEqual(float(self.model.body_mass.sum()), self.fk.total, places=9)
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        transforms = self.fk.fk({})
        com = sum(m*(transforms[n] @ o)[:3] for n, m, o in self.fk.masses)/self.fk.total
        np.testing.assert_allclose(data.subtree_com[0], com, atol=1e-9)


if __name__ == '__main__':
    unittest.main()


@unittest.skipIf(mujoco is None, 'mujoco not installed')
class AnkleRollContractTest(unittest.TestCase):
    """12-DOF variant (user decision 2026-09-29): ankle roll added, nothing else changed."""

    def test_twelve_actuators_same_mass_and_com(self):
        import xml.etree.ElementTree as ET
        m10 = mujoco.MjModel.from_xml_path(str(ROOT/'sim/raptor_digitigrade.xml'))
        m12 = mujoco.MjModel.from_xml_path(str(ROOT/'sim/raptor_digitigrade_ankleroll.xml'))
        names = [m12.actuator(i).name for i in range(m12.nu)]
        self.assertEqual(len(names), 12)
        self.assertEqual([n for n in names if 'ankle_roll' not in n], [m10.actuator(i).name for i in range(m10.nu)])
        self.assertAlmostEqual(sum(m12.body_mass), sum(m10.body_mass), places=6)
        d10, d12 = mujoco.MjData(m10), mujoco.MjData(m12)
        mujoco.mj_forward(m10, d10); mujoco.mj_forward(m12, d12)
        np.testing.assert_allclose(d12.subtree_com[1], d10.subtree_com[1], atol=1e-6)
        urdf = ET.fromstring((ROOT/'sim/raptor_digitigrade_ankleroll.urdf').read_text())
        for side in ('left', 'right'):
            joint = next(j for j in urdf.findall('joint') if j.get('name') == f'{side}_ankle_roll_joint')
            self.assertEqual(joint.find('axis').get('xyz'), '1 0 0')
            self.assertEqual(joint.find('child').get('link'), f'{side}_foot_link')
