"""R-02 contract: design table, Xacro-generated MJCF and the 12-DOF interface agree (DESIGN_R02.ko.md)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'modeling'))
try:
    import mujoco
    import numpy as np
    import design_r02 as D
except ImportError:  # .venv-ai has no simulator stack; use .venv-sim
    mujoco = None

YAML = ROOT/'src/raptor_description/config/r02_design.yaml'


@unittest.skipIf(mujoco is None, 'mujoco not installed')
class R02ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = json.loads(YAML.read_text())
        cls.m = mujoco.MjModel.from_xml_path(str(ROOT/'sim/raptor_r02.xml'))
        cls.ma = mujoco.MjModel.from_xml_path(str(ROOT/'sim/raptor_r02_achilles.xml'))

    def test_yaml_is_current(self):
        self.assertEqual(json.loads(json.dumps(D.link_table(D.Params()))), self.table,
                         'r02_design.yaml is stale: run modeling/build_r02.py')

    def test_mass_matches_design(self):
        self.assertAlmostEqual(self.m.body_subtreemass[0], self.table['total_mass'], delta=.01)

    def test_same_active_interface_as_12_dof(self):
        ref = mujoco.MjModel.from_xml_path(str(ROOT/'sim/raptor_digitigrade_ankleroll.xml'))
        names = lambda m: [m.actuator(i).name for i in range(m.nu)]
        self.assertEqual(names(self.m), names(ref))
        for i in range(self.m.nu):
            j = self.table['joints'][self.m.actuator(i).name.removeprefix('left_').removeprefix('right_').removesuffix('_joint')]
            self.assertAlmostEqual(self.m.actuator_forcerange[i, 1], j['effort'])
            self.assertAlmostEqual(self.m.actuator_ctrlrange[i, 1], j['velocity'])

    def nominal(self, m):
        d = mujoco.MjData(m)
        for side in ('left', 'right'):
            for joint, v in self.table['nominal_pose'].items():
                d.qpos[m.jnt_qposadr[m.joint(f'{side}_{joint}_joint').id]] = v
        mujoco.mj_forward(m, d)
        return d

    def test_toes_flat_and_com_over_toes(self):
        m, d = self.m, self.nominal(self.m)
        for side, n in (('left', 2), ('left', 1), ('right', 2), ('right', 3)):  # digits III and IV lie flat
            x_axis = d.xmat[m.body(f'{side}_toe_{n}_proximal_link').id].reshape(3, 3)[:, 0]
            self.assertLess(abs(x_axis[2]), 1e-3)
        mtp = d.xpos[m.body('left_foot_link').id]
        com = d.subtree_com[0]
        self.assertGreater(com[0], mtp[0])                                   # in front of the toe base
        self.assertLess(com[0], mtp[0] + self.table['toes']['III']['length'])  # behind the digit III tip

    def test_toe_springs_and_achilles(self):
        for side, n, digit in (('left', 2, 'III'), ('left', 1, 'IV'), ('left', 3, 'II'), ('right', 3, 'IV')):
            j = self.m.joint(f'{side}_toe_{n}_proximal_joint').id
            self.assertAlmostEqual(self.m.jnt_stiffness[j], self.table['toes'][digit]['k_proximal'])
        a = self.table['achilles']
        j = self.ma.joint('left_ankle_pitch_joint').id
        self.assertAlmostEqual(self.ma.jnt_stiffness[j], a['stiffness'])
        self.assertAlmostEqual(self.ma.qpos_spring[self.ma.jnt_qposadr[j]],
                               self.table['nominal_pose']['ankle_pitch'] + a['preload']/a['stiffness'], places=4)
        self.assertEqual(self.m.jnt_stiffness[self.m.joint('left_ankle_pitch_joint').id], 0.)


if __name__ == '__main__':
    unittest.main()
