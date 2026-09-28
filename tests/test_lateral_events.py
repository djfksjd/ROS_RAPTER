"""Foot-event detection used by experiment F (sim/lateral_experiments.py)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim'))
from lateral_experiments import step_events, transitions  # noqa: E402

W = 100.


class StepEventsTest(unittest.TestCase):
    def setUp(self):
        self.t = np.round(np.arange(-.5, 2., .01), 3)
        self.fz = {'left': np.full(len(self.t), 50.), 'right': np.full(len(self.t), 50.)}
        off = (self.t >= .05) & (self.t < .30)  # left foot in the air for 0.25 s
        self.fz['left'][off], self.fz['right'][off] = 0., 100.

    def test_liftoff_and_touchdown(self):
        ev = step_events(self.t, self.fz, W, [0.])[0]
        self.assertTrue(ev['settled'])
        self.assertEqual(ev['lifted'], 'left')
        self.assertAlmostEqual(ev['liftoff_s'], .05)
        self.assertAlmostEqual(ev['single_support_s'], .25)
        self.assertEqual([x[1:] for x in ev['transitions_1p5s']], [('left', 'off'), ('left', 'on')])

    def test_duplicate_timestamps_do_not_break_the_hold_length(self):
        t = np.repeat(self.t, 2)  # Gazebo logs can repeat a stamp
        fz = {k: np.repeat(v, 2) for k, v in self.fz.items()}
        self.assertAlmostEqual(step_events(t, fz, W, [0.])[0]['single_support_s'], .25)

    def test_event_after_data_and_unsettled_start(self):
        ev = step_events(self.t, self.fz, W, [5.])[0]
        self.assertFalse(ev['settled'])
        self.assertNotIn('lifted', ev)
        self.assertFalse(step_events(self.t, self.fz, W, [.1])[0]['settled'])

    def test_transitions_hysteresis(self):
        fz = {'left': np.array([50., 10., 3., 10., 25.]), 'right': np.full(5, 50.)}  # 10 N is between thresholds
        self.assertEqual(transitions(np.arange(5)*.01, fz, W, 0., 1.), [(.02, 'left', 'off'), (.04, 'left', 'on')])


if __name__ == '__main__':
    unittest.main()
