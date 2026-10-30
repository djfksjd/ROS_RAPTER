"""Model rev 3 (stage 0, evidence 186-187): R-03 MJCF with hip yaw, one axis config (sim/rl/rev3_b.json) for the env and
the realism layer, vendor T-N curve cap, and the rev 2 path left unchanged."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sim'/'rl'))
sys.path.insert(0, str(ROOT/'sim'))
try:
    import gymnasium  # noqa: F401
    import mujoco  # noqa: F401
    import numpy as np
except ImportError:  # .venv-ai has no simulator stack; use .venv-sim
    mujoco = None

KW = dict(arch8=True, real_mass=True, randomize=False, seed=1)
REALISM = str(ROOT/'sim/rl/realism_robstride_rev3.json')


@unittest.skipIf(mujoco is None, 'mujoco/gymnasium not installed')
class Rev3ModelTest(unittest.TestCase):
    def test_r03_xml_is_generated(self):
        import build_r03
        text = (ROOT/'sim/raptor_r03.xml').read_text().split('\n', 1)[1]
        self.assertEqual(text.strip(), build_r03.build().to_xml().strip())

    def test_axes_actions_mass(self):
        from run_env import RunEnv, KNEE_BAND
        e = RunEnv(**KW, model_rev=3)
        self.assertEqual(len(e.active), 14)
        self.assertEqual(e.action_space.shape, (10,))  # 4 per leg (yaw, roll, pitch, knee) + 2 tail
        self.assertEqual(sorted(e.active[i] for i in e.policy_idx if 'left' in e.active[i]),
                         ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_pitch_joint'])
        self.assertAlmostEqual(e.mass, 15.1, delta=.1)  # evidence 186 fixed point
        self.assertLessEqual(e.knee_band[0], KNEE_BAND[0])
        self.assertGreaterEqual(e.knee_band[1], KNEE_BAND[1])

    def test_realism_matches_axis_config(self):
        ax = json.loads((ROOT/'sim/rl/rev3_b.json').read_text())['axes']
        rz = json.loads(Path(REALISM).read_text())['motor']
        self.assertTrue(rz['tn_curve'])
        self.assertIn('RS04', rz['spec_override'])
        for jt, a in ax.items():
            self.assertEqual(rz['assign'][jt], {'motor': a['module'], 'external_ratio': a['ratio'], 'link_efficiency': a['link_efficiency']})

    def test_tn_curve_caps_drive_torque(self):
        from run_env import RunEnv
        e = RunEnv(**KW, model_rev=3, realism=REALISM)
        e.reset(seed=1)
        i = e.active.index('left_hip_pitch_joint')
        j = list(e.rz.idx).index(i)
        e.data.qvel[e.v_adr[i]] = 15.  # joint rad/s; module output 15 x 0.85 = 12.75 rad/s (122 rpm)
        lo, hi = e.rz.motor_limits(e, np.full(len(e.active), -1e3), np.full(len(e.active), 1e3))
        w, t = e.rz.tn[j]
        cap = np.interp(12.75, w, t)*.85*e.rz.eta[j]
        self.assertLessEqual(hi[i], cap+1e-6)
        self.assertLess(cap, 120*.85*.9)  # below the peak: the curve starts dropping at 95 rpm

    def test_mirror_flips_hip_yaw(self):
        from mirror_probe import ObservationReflection, joint_reflection
        from run_env import RunEnv
        e = RunEnv(**KW, model_rev=3)
        o, _ = e.reset(seed=1)
        idx, sign = joint_reflection(e.active)
        ly, ry = e.active.index('left_hip_yaw_joint'), e.active.index('right_hip_yaw_joint')
        self.assertEqual(idx[ly], ry)
        self.assertEqual(sign[ly], -1.)
        r = ObservationReflection(e.active, e.policy_idx, e.q0, e.obs_vel, e.obs_contact)
        x = o + np.random.default_rng(0).normal(0, .1, o.shape)
        np.testing.assert_allclose(r.observation(r.observation(x)), x, atol=1e-12)

    def test_rev2_unchanged(self):
        from run_env import RunEnv
        e = RunEnv(**KW, model_rev=2)
        self.assertAlmostEqual(e.mass, 11.2912, places=3)
        self.assertEqual(e.action_space.shape, (8,))
        self.assertNotIn('left_hip_yaw_joint', e.active)


if __name__ == '__main__':
    unittest.main()
