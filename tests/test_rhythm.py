# tests/test_rhythm.py
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

from rhythm import RollPhaseOscillator, wrap


class TestRollPhaseOscillator(unittest.TestCase):
    def test_wrap(self):
        self.assertGreaterEqual(wrap(3 * np.pi), -np.pi)
        self.assertLess(wrap(3 * np.pi), np.pi)
        self.assertAlmostEqual(wrap(3 * np.pi), -np.pi)
        self.assertAlmostEqual(wrap(-3.5 * np.pi), 0.5 * np.pi)

    def test_open_loop(self):
        osc = RollPhaseOscillator(gain=0, amplitude_feedback=False)
        n = 5000
        for _ in range(n):
            osc.step(0.0)
        self.assertAlmostEqual(osc.theta, n * osc.dt * osc.omega0, delta=1e-9)

    def test_correction_limit(self):
        for roll in (-1.0, 1.0):
            osc = RollPhaseOscillator(
                gain=1e6, correction_limit=0.5,
                enable_after=0.0, amplitude_feedback=False,
            )
            for _ in range(100):
                previous = osc.theta
                osc.step(roll)
                advance = osc.theta - previous
                self.assertLessEqual(
                    advance, osc.dt * (osc.omega0 + osc.correction_limit) + 1e-12
                )
                self.assertGreaterEqual(
                    advance, osc.dt * (osc.omega0 - osc.correction_limit) - 1e-12
                )

    def test_amplitude_limits(self):
        for huge in (False, True):
            osc = RollPhaseOscillator(gain=0)
            initial = osc.amplitude
            steps = int(20 / 2 / osc.dt)
            for i in range(steps):
                roll = np.sin(osc.omega0 * i * osc.dt) if huge else 0.0
                osc.step(roll)
                self.assertGreaterEqual(osc.amplitude, osc.amplitude_limits[0])
                self.assertLessEqual(osc.amplitude, osc.amplitude_limits[1])
            self.assertGreaterEqual(osc.cycles, 19)
            if huge:
                self.assertLess(osc.amplitude, initial)
            else:
                self.assertGreater(osc.amplitude, initial)

    def test_phase_lock(self):
        osc = RollPhaseOscillator(
            gain=5, phase_offset=0, amplitude_feedback=False
        )
        for i in range(5000):
            t = i * osc.dt
            osc.step(0.1 * np.sin(osc.omega0 * t + 1.0))
        target = osc.omega0 * t + 1.0
        self.assertLess(abs(wrap(osc.theta - target)), 0.35)


if __name__ == "__main__":
    unittest.main()
