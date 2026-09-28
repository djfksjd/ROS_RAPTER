"""roll_diagnosis metric helpers on synthetic signals."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim'))
import roll_diagnosis as rd  # noqa: E402


class MetricTest(unittest.TestCase):
    def test_band_amplitude_of_a_sine(self):
        t = np.arange(0, 20, .01)
        amp, peak = rd.band_amplitude(t, .05*np.sin(2*np.pi*.8*t), 0, 20, .3, 1.6)
        self.assertAlmostEqual(peak, .8, delta=.06)
        self.assertAlmostEqual(amp, .05, delta=.01)

    def test_growth_needs_two_cycles(self):
        t = np.arange(2, 2.74, .01)
        self.assertEqual(rd.step_growth(t, np.sin(t), 2, 3, 1.0), (None, None, []))

    def test_growth_of_a_steady_oscillation_is_one(self):
        t = np.arange(0, 20, .01)
        median, _, _ = rd.step_growth(t, .03*np.sin(2*np.pi*2.5*t), 2, 18, 2.5)
        self.assertAlmostEqual(median, 1., delta=.02)

    def test_single_support_median(self):
        t = np.arange(0, 2, .01)
        fl = np.where((t % .5) < .25, 150., 20.)  # left carries the load half of each 0.5 s
        fr = np.where((t % .5) < .25, 20., 150.)
        self.assertAlmostEqual(rd.single_support_median(t, fl, fr, 170.), .25, delta=.011)


if __name__ == '__main__':
    unittest.main()
