"""rock_law.targets must follow the sim/rock_probe.py law (lift = 0)."""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/raptor_control/scripts'))
from rock_law import crouch_pose, tail_targets, targets  # noqa: E402

JOINTS = [f'{s}_{j}_joint' for s in ('left', 'right') for j in ('hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')]
JOINTS += ['tail_yaw_joint', 'tail_pitch_joint']
POSE = crouch_pose(JOINTS, -.10, .5)


def reference(a, stride, window=.35, ramp=1., f=2.):
    """Inline copy of the rock_probe.py stride branch (lift = 0, smooth_swing False)."""
    out = {}
    cycle = (a*f) % 1
    for side, center in (('left', .28), ('right', .78)):
        offset = (cycle-center+.5) % 1-.5
        u = (offset+window/2)/window
        swing = -stride*(u-.5) if 0 <= u <= 1 else -stride*(.5-((offset-window/2) % 1)/(1-window))
        out[side] = swing*min(1., max(0., a-ramp))
    return out


class RockLawTest(unittest.TestCase):
    def test_before_start_is_the_crouch(self):
        self.assertEqual(targets(POSE, -.1, .08, 2., 1., stride=.05), POSE)

    def test_hip_rolls_mirror(self):
        t = targets(POSE, 1.125, .08, 2., 1.)
        self.assertAlmostEqual(t['left_hip_roll_joint'], -.08*math.sin(2*math.pi*2*1.125))
        self.assertAlmostEqual(t['right_hip_roll_joint'], -t['left_hip_roll_joint'])

    def test_stride_matches_rock_probe_and_keeps_sole_level(self):
        for k in range(200):
            a = 1.5+k*.013
            t, ref = targets(POSE, a, .08, 2., 1., stride=.05), reference(a, .05)
            for side in ('left', 'right'):
                swing = t[f'{side}_hip_pitch_joint']-POSE[f'{side}_hip_pitch_joint']
                self.assertAlmostEqual(swing, ref[side])
                level = sum(t[f'{side}_{j}_joint'] for j in ('hip_pitch', 'knee_pitch', 'ankle_pitch'))
                self.assertAlmostEqual(level, 0.)

    def test_pitch_feedback_is_clipped(self):
        t = targets(POSE, 2., .08, 2., 1., pitch=1., feedback=(.5, .05, .15))
        self.assertAlmostEqual(t['left_ankle_pitch_joint']-POSE['left_ankle_pitch_joint'], .15)

    def test_tail_targets_signs_and_limits(self):
        t = tail_targets(.05, roll=.1, pitch=.02, pitch_rate=.1, gains=(1.5, 2.5, 2., .1))
        self.assertAlmostEqual(t['tail_yaw_joint'], 1.5*.05+2.5*.1)
        self.assertAlmostEqual(t['tail_pitch_joint'], 2*.02+.1*.1)
        big = tail_targets(1., roll=1., pitch=1., gains=(3., 3., 3., 0.))
        self.assertEqual((big['tail_yaw_joint'], big['tail_pitch_joint']), (.6, .4))
        self.assertEqual(tail_targets(.05), {'tail_yaw_joint': 0., 'tail_pitch_joint': 0.})


if __name__ == '__main__':
    unittest.main()
