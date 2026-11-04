"""Stage-4 thresholds, heading wrapping and complete-suite adoption checks."""
import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim/rl'))
from gate_judge import judge, turn_suite_status
from turn_rules import TURN_CRITERIA


def turn_row(speed=6., sign=1.):
    yaw = sign*TURN_CRITERIA[speed]['yaw']
    joint = dict(rms_nm=5., peak_nm=16., peak_rad_s=7., at_limit=0.)
    episodes = [dict(seed=s, t_fall=None, v_mean=speed, yaw_rate_mean=yaw,
                     radius_m=abs(speed/yaw), t_90_s=math.pi/(2*abs(yaw)),
                     heading_change_rad=sign*3., load_samples=10000,
                     joints={name: dict(joint) for name in ('left_hip_yaw_joint', 'right_hip_yaw_joint')})
                for s in range(7300, 7304)]
    return dict(gate='turn', cond='moving_turn', cmd=speed, yaw_cmd=yaw, quick=False,
                n=4, falls=0, episodes=episodes)


def complete_suite():
    rows = [turn_row(v, sign) for v in TURN_CRITERIA for sign in (1., -1.)]
    def add(gate, cond, n, **extra):
        e = [dict(seed=i, t_fall=None, v_mean=extra.get('cmd', 0), min_derate=1.,
                  yaw_rate_mean=extra.get('cmd', 0), drift_m=.1, t_reach_0p8=[1, 1],
                  speed_2s_after_stop=[0, 0], switches=2) for i in range(n)]
        rows.append(dict(gate=gate, cond=cond, n=n, falls=0, episodes=e, drift_max_m=.1, **extra))
    for v in (2., 4., 6.): add('turn', 'run_regression', 4, cmd=v)
    add('stand', 'hold', 8); add('stand', 'hold_dr', 24)
    for d in ('front', 'back', 'left', 'right'): add('stand', 'push', 8, dv=.3, dir=d)
    for v in (.3, .6, 1., 1.5): add('walk', 'track', 4, cmd=v)
    for w in (.5, -.5): add('walk', 'turn', 4, cmd=w)
    add('walk', 'stopstart', 4); add('walk', 'track_dr', 24, cmd=1.)
    add('switch', 'switch_ramp', 4); add('switch', 'switch_ramp_dr', 8)
    return rows


class TurnGateTest(unittest.TestCase):
    def test_both_signs_and_all_speeds(self):
        for v in TURN_CRITERIA:
            for sign in (1., -1.):
                self.assertTrue(judge(turn_row(v, sign))[0])

    def test_reject_each_bad_metric(self):
        for key, value in [('v_mean', 4.9), ('yaw_rate_mean', .32), ('yaw_rate_mean', -.5),
                           ('radius_m', 21.01), ('t_90_s', 6.01), ('t_90_s', None),
                           ('heading_change_rad', 1.), ('t_fall', 12.), ('v_mean', float('nan'))]:
            row = turn_row(); row['episodes'][0][key] = value
            with self.subTest(key=key, value=value): self.assertFalse(judge(row)[0])
        row = turn_row(); row['episodes'] = []
        self.assertFalse(judge(row)[0])

    def test_complete_suite_and_missing_regression(self):
        rows = complete_suite()
        self.assertTrue(turn_suite_status(rows)[0])
        self.assertFalse(turn_suite_status([])[0])
        for i in range(len(rows)):
            self.assertFalse(turn_suite_status(rows[:i]+rows[i+1:])[0])
        self.assertFalse(turn_suite_status(rows+[copy.deepcopy(rows[0])])[0])
        for key, value in [('quick', True), ('n', 2), ('falls', 1)]:
            changed = copy.deepcopy(rows); changed[0][key] = value
            self.assertFalse(turn_suite_status(changed)[0])

    def test_load_and_holdout_required(self):
        for key, value in [('seed', 7200), ('load_samples', 500), ('joints', {})]:
            rows = complete_suite(); rows[0]['episodes'][0][key] = value
            self.assertFalse(turn_suite_status(rows)[0])
        rows = complete_suite(); rows[-1]['falls'] = 2
        rows[-1]['episodes'][0]['t_fall'] = 5.; rows[-1]['episodes'][1]['t_fall'] = 8.
        self.assertFalse(turn_suite_status(rows)[0])

    def test_heading_wrap(self):
        try:
            from turn_eval import heading_delta
        except ImportError:
            self.skipTest('sim stack not installed')
        self.assertAlmostEqual(heading_delta(math.pi-.1, -math.pi+.1), .2)
        self.assertAlmostEqual(heading_delta(-math.pi+.1, math.pi-.1), -.2)


if __name__ == '__main__':
    unittest.main()
